#!/usr/bin/env python3
# Standalone CI script (call graph), not urwid library code.
# ruff: noqa: C416, D101, D102, D103, D107, D205, D209, EXE001, PERF401, PLC0207, PLW2901, S603, S607, UP035, UP037, UP045
"""Static call-graph extractor for Python source trees (stdlib only).

Pipeline
  1. Parse every ``.py`` file with :mod:`ast`.
  2. Collect definition nodes: modules, classes, functions, methods
     (nested functions get ``outer.<locals>.inner`` ids, like ``__qualname__``).
  3. Build symbol tables: imports (absolute + relative), module-level aliases,
     class bases, and light type information (parameter / return annotations,
     ``x = Cls(...)`` assignments, ``self.attr = ...`` in methods).
  4. Visit every ``ast.Call`` in each scope and resolve its callee to a
     definition id, an external dotted name, a builtin, or "unresolved".
  5. Write ``graph.json`` (nodes + aggregated edges + stats).

This is static analysis: dynamic dispatch, monkey-patching, getattr() and
untyped variables can't be resolved; such calls are counted as unresolved.

Usage:
    python tools/callgraph/callgraph.py httpx -o graph.json
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

__all__ = (
    "Ctx",
    "Def",
    "ModuleInfo",
    "Project",
    "bound_names",
    "dotted",
    "git_commit",
    "graph_edges",
    "is_super_call",
    "iter_py_files",
    "literal_type",
    "main",
    "markdown_summary",
    "module_name_for",
    "walk_scope",
)


__version__ = "0.4.0"

BUILTIN_NAMES = frozenset(dir(builtins))
DEFAULT_EXCLUDES = frozenset({".git", ".venv", "venv", "__pycache__", "build", "dist", "node_modules", ".tox"})
FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
SCOPE_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
# graph.json is networkx node_link data (swegemma format): edge "type" values.
EDGE_TYPES = {"call": "calls", "instantiate": "instantiates"}
PROPERTY_DECORATORS = {"property", "cached_property", "functools.cached_property"}
# Return types of a few common external factories (extend as needed).
KNOWN_RETURNS = {
    "logging.getLogger": "logging.Logger",
    "ssl.create_default_context": "ssl.SSLContext",
}
LITERAL_TYPES = (
    (ast.JoinedStr, "builtins.str"),
    ((ast.Dict, ast.DictComp), "builtins.dict"),
    ((ast.List, ast.ListComp), "builtins.list"),
    ((ast.Set, ast.SetComp), "builtins.set"),
    (ast.Tuple, "builtins.tuple"),
)


def literal_type(expr: ast.AST) -> Optional[str]:
    if isinstance(expr, ast.Constant):
        if isinstance(expr.value, str):
            return "builtins.str"
        if isinstance(expr.value, bytes):
            return "builtins.bytes"
        return None
    for types, name in LITERAL_TYPES:
        if isinstance(expr, types):
            return name
    return None


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass
class Def:
    id: str
    kind: str  # module | class | function | method
    name: str
    module: str
    file: str
    lineno: int
    end_lineno: int
    parent: Optional[str]
    node: ast.AST
    extra_nodes: list = field(default_factory=list)  # redefinitions (setters, overloads)
    decorators: list = field(default_factory=list)
    is_async: bool = False
    local_defs: dict = field(default_factory=dict)  # name -> def id (direct children)
    bases: list = field(default_factory=list)  # resolved base ids / dotted names


@dataclass
class ModuleInfo:
    name: str
    file: str
    is_pkg: bool
    tree: ast.Module
    imports: dict = field(default_factory=dict)  # local name -> fully-qualified name
    defs: dict = field(default_factory=dict)  # top-level name -> def id
    aliases: dict = field(default_factory=dict)  # top-level name -> value expr


@dataclass
class Ctx:
    """Resolution context for code inside one definition."""

    d: Def
    mi: ModuleInfo
    cls: Optional[str] = None  # class id when inside a method (for self/super)
    parent: Optional["Ctx"] = None
    func_chain: list = field(default_factory=list)  # enclosing function Defs, innermost first
    local_names: set = field(default_factory=set)
    local_types: dict = field(default_factory=dict)  # var -> type id / dotted name
    local_classes: dict = field(default_factory=dict)  # `cls` param -> class id

    def is_local(self, name: str) -> bool:
        c: Optional[Ctx] = self
        while c is not None:
            if name in c.local_names:
                return True
            c = c.parent
        return False

    def var_type(self, name: str) -> Optional[str]:
        c: Optional[Ctx] = self
        while c is not None:
            if name in c.local_types:
                return c.local_types[name]
            if name in c.local_names:
                return None  # shadowed by an untyped local
            c = c.parent
        return None

    def local_class(self, name: str) -> Optional[str]:
        c: Optional[Ctx] = self
        while c is not None:
            if name in c.local_classes:
                return c.local_classes[name]
            c = c.parent
        return None


# --------------------------------------------------------------------------
# AST helpers
# --------------------------------------------------------------------------


def walk_scope(node: ast.AST) -> Iterator[ast.AST]:
    """Yield nodes evaluated in ``node``'s own scope.

    Nested function/class definitions are yielded but their bodies are not
    entered; their decorators, defaults and bases are, because Python
    evaluates those in the enclosing scope.
    """
    if isinstance(node, SCOPE_TYPES):
        stack = list(node.body)
    else:
        stack = list(ast.iter_child_nodes(node))
    stack.reverse()
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, FUNC_TYPES):
            outer = list(n.decorator_list) + list(n.args.defaults)
            outer += [d for d in n.args.kw_defaults if d is not None]
            stack.extend(reversed(outer))
        elif isinstance(n, ast.ClassDef):
            outer = list(n.decorator_list) + list(n.bases) + [k.value for k in n.keywords]
            stack.extend(reversed(outer))
        else:
            stack.extend(reversed(list(ast.iter_child_nodes(n))))


def dotted(expr: ast.AST) -> Optional[list]:
    """``a.b.c`` -> ['a', 'b', 'c']; anything else -> None."""
    parts = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name):
        parts.append(expr.id)
        return parts[::-1]
    return None


def bound_names(target: ast.AST) -> Iterator[str]:
    for n in ast.walk(target):
        if isinstance(n, ast.Name):
            yield n.id


def is_super_call(expr: ast.AST) -> bool:
    return isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "super"


def module_name_for(path: Path, root: Path) -> tuple:
    rel = path.resolve().relative_to(root.resolve())
    parts = list(rel.with_suffix("").parts)
    is_pkg = parts[-1] == "__init__"
    if is_pkg:
        parts = parts[:-1]
    return ".".join(parts) or rel.stem, is_pkg


def iter_py_files(paths: list, excludes: frozenset) -> Iterator[Path]:
    seen = set()
    for p in map(Path, paths):
        files = [p] if p.is_file() else sorted(p.rglob("*.py")) if p.is_dir() else []
        for f in files:
            if f.suffix != ".py" or any(part in excludes for part in f.parts):
                continue
            key = f.resolve()
            if key not in seen:
                seen.add(key)
                yield f


# --------------------------------------------------------------------------
# Project analysis
# --------------------------------------------------------------------------


class Project:
    def __init__(self, root: Path):
        self.root = root
        self.modules: dict = {}
        self.defs: dict = {}
        self.top_packages: set = set()
        self.class_attr_types: dict = {}
        self.module_var_types: dict = {}
        self.warnings: list = []
        self.sources: dict = {}  # module -> source lines (keepends)
        self._ctx_cache: dict = {}
        self._ret_cache: dict = {}

    # ---- loading -----------------------------------------------------------

    def load(self, files: list) -> None:
        for f in files:
            try:
                src = f.read_text(encoding="utf-8")
                tree = ast.parse(src, filename=str(f))
            except (SyntaxError, UnicodeDecodeError) as exc:
                self.warnings.append(f"skip {f}: {exc}")
                continue
            name, is_pkg = module_name_for(f, self.root)
            rel = f.resolve().relative_to(self.root.resolve()).as_posix()
            mi = ModuleInfo(name=name, file=rel, is_pkg=is_pkg, tree=tree)
            self.modules[name] = mi
            self.sources[name] = src.splitlines(keepends=True)
            self.top_packages.add(name.split(".")[0])
            mdef = Def(
                id=name,
                kind="module",
                name=name.rsplit(".", 1)[-1],
                module=name,
                file=rel,
                lineno=1,
                end_lineno=len(src.splitlines()) or 1,
                parent=None,
                node=tree,
            )
            self.defs[name] = mdef
        for mi in self.modules.values():
            self._collect_imports(mi)
            self._collect_defs(mi, mi.tree, self.defs[mi.name])
            self._collect_aliases(mi)
        for d in self.defs.values():
            if d.kind == "class":
                d.bases = self._resolve_bases(d)

    def _collect_imports(self, mi: ModuleInfo) -> None:
        package = mi.name if mi.is_pkg else mi.name.rpartition(".")[0]
        for n in ast.walk(mi.tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    if a.asname:
                        mi.imports[a.asname] = a.name
                    else:
                        head = a.name.split(".")[0]
                        mi.imports[head] = head
            elif isinstance(n, ast.ImportFrom):
                if n.level:
                    base_parts = package.split(".") if package else []
                    if n.level > 1:
                        base_parts = base_parts[: len(base_parts) - (n.level - 1)]
                    base = ".".join(base_parts)
                    src = ".".join(p for p in (base, n.module) if p)
                else:
                    src = n.module or ""
                for a in n.names:
                    if a.name == "*":
                        continue  # star imports resolved lazily in resolve_fq
                    mi.imports[a.asname or a.name] = f"{src}.{a.name}" if src else a.name

    def _collect_defs(self, mi: ModuleInfo, scope_node: ast.AST, owner: Def) -> None:
        for n in walk_scope(scope_node):
            if not isinstance(n, SCOPE_TYPES):
                continue
            if owner.kind == "module":
                kind = "class" if isinstance(n, ast.ClassDef) else "function"
                did = f"{owner.id}.{n.name}"
            elif owner.kind == "class":
                kind = "class" if isinstance(n, ast.ClassDef) else "method"
                did = f"{owner.id}.{n.name}"
            else:
                kind = "class" if isinstance(n, ast.ClassDef) else "function"
                did = f"{owner.id}.<locals>.{n.name}"
            if did in self.defs:
                # property setter / overload / conditional redefinition:
                # attribute its calls to the first definition.
                existing = self.defs[did]
                existing.extra_nodes.append(n)
                existing.end_lineno = max(existing.end_lineno, n.end_lineno or n.lineno)
                self._collect_defs(mi, n, existing)
                continue
            d = Def(
                id=did,
                kind=kind,
                name=n.name,
                module=mi.name,
                file=mi.file,
                lineno=n.lineno,
                end_lineno=n.end_lineno or n.lineno,
                parent=owner.id,
                node=n,
                decorators=[ast.unparse(x) for x in n.decorator_list],
                is_async=isinstance(n, ast.AsyncFunctionDef),
            )
            self.defs[did] = d
            owner.local_defs[n.name] = did
            if owner.kind == "module":
                mi.defs[n.name] = did
            self._collect_defs(mi, n, d)

    def _collect_aliases(self, mi: ModuleInfo) -> None:
        for n in walk_scope(mi.tree):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                mi.aliases.setdefault(n.targets[0].id, n.value)
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
                mi.aliases.setdefault(n.target.id, n.value)

    def _resolve_bases(self, d: Def) -> list:
        mctx = self.module_ctx(d.module)
        out = []
        for b in d.node.bases:  # type: ignore[attr-defined]
            if isinstance(b, ast.Subscript):  # Generic[T], Base[X]
                b = b.value
            parts = dotted(b)
            if not parts:
                continue
            sym = self.resolve_name(mctx, parts[0], parts[1:])
            if sym:
                out.append(sym[0])
        return out

    # ---- symbol resolution ------------------------------------------------

    def in_project(self, fq: str) -> bool:
        return fq.split(".")[0] in self.top_packages

    def module_expand(self, mi: ModuleInfo, parts: list) -> Optional[str]:
        """Expand a dotted name to a fully-qualified name using module scope."""
        head, rest = parts[0], parts[1:]
        if head in mi.defs:
            base = mi.defs[head]
        elif head in mi.imports:
            base = mi.imports[head]
        elif head in mi.aliases:
            base = f"{mi.name}.{head}"
        elif head in BUILTIN_NAMES:
            base = f"builtins.{head}"
        else:
            return None
        return ".".join([base, *rest])

    def resolve_fq(self, fq: str, depth: int = 0) -> Optional[str]:
        """Map a fully-qualified name to a project def id, following re-exports."""
        if fq in self.defs:
            return fq
        if depth > 12:
            return None
        parts = fq.split(".")
        for i in range(len(parts) - 1, 0, -1):
            prefix, rest = ".".join(parts[:i]), parts[i:]
            d = self.defs.get(prefix)
            if d is None:
                continue
            if d.kind == "module":
                mi = self.modules[prefix]
                head = rest[0]
                if head in mi.imports:
                    return self.resolve_fq(".".join([mi.imports[head], *rest[1:]]), depth + 1)
                if head in mi.aliases:
                    vparts = dotted(mi.aliases[head])
                    if vparts:
                        tgt = self.module_expand(mi, vparts)
                        if tgt and tgt != fq:
                            return self.resolve_fq(".".join([tgt, *rest[1:]]), depth + 1)
                for star in self._star_sources(mi):
                    r = self.resolve_fq(".".join([star, *rest]), depth + 1)
                    if r:
                        return r
                return None
            if d.kind == "class":
                m = self.lookup_member(prefix, rest[0])
                if m is None:
                    return None
                return m if len(rest) == 1 else self.resolve_fq(".".join([m, *rest[1:]]), depth + 1)
            return None
        return None

    def _star_sources(self, mi: ModuleInfo) -> list:
        package = mi.name if mi.is_pkg else mi.name.rpartition(".")[0]
        out = []
        for n in mi.tree.body:
            if isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names):
                if n.level:
                    bp = package.split(".")
                    if n.level > 1:
                        bp = bp[: len(bp) - (n.level - 1)]
                    out.append(".".join(p for p in (".".join(bp), n.module) if p))
                elif n.module:
                    out.append(n.module)
        return out

    def resolve_name(self, ctx: Ctx, head: str, rest: list = ()) -> Optional[tuple]:
        """Resolve ``head.rest...`` seen in ``ctx`` -> (target, how) or None.

        target is a project def id, or a dotted external/builtin name.
        None means it points into the project but can't be pinned down.
        """
        rest = list(rest)
        for fd in ctx.func_chain:
            if head in fd.local_defs:
                return self._descend(fd.local_defs[head], rest, "local")
        mi = ctx.mi
        if head in mi.defs:
            return self._descend(mi.defs[head], rest, "module")
        if head in mi.imports:
            fq, how = ".".join([mi.imports[head], *rest]), "import"
        elif head in mi.aliases:
            vparts = dotted(mi.aliases[head])
            if not vparts or vparts[0] == head:
                return None
            return self.resolve_name(self.module_ctx(mi.name), vparts[0], vparts[1:] + rest)
        elif head in BUILTIN_NAMES:
            return (".".join(["builtins", head, *rest]), "builtin")
        else:
            star = [s for s in self._star_sources(mi)]
            for s in star:
                r = self.resolve_fq(".".join([s, head, *rest]))
                if r:
                    return (r, "import")
            return None
        target = self.resolve_fq(fq)
        if target:
            return (target, how)
        if self.in_project(fq):
            return None
        return (fq, "external")

    def _descend(self, def_id: str, rest: list, how: str) -> Optional[tuple]:
        if not rest:
            return (def_id, how)
        t = self.resolve_fq(".".join([def_id, *rest]))
        return (t, how) if t else None

    def lookup_member(self, cls_id: str, attr: str, seen: Optional[set] = None) -> Optional[str]:
        seen = set() if seen is None else seen
        if cls_id in seen or cls_id not in self.defs:
            return None
        seen.add(cls_id)
        cand = f"{cls_id}.{attr}"
        if cand in self.defs:
            return cand
        for b in self.defs[cls_id].bases:
            if b in self.defs and self.defs[b].kind == "class":
                r = self.lookup_member(b, attr, seen)
                if r:
                    return r
        return None

    def external_base(self, cls_id: str, seen: Optional[set] = None) -> Optional[str]:
        seen = set() if seen is None else seen
        if cls_id in seen:
            return None
        seen.add(cls_id)
        for b in self.defs[cls_id].bases:
            if b in self.defs:
                r = self.external_base(b, seen)
                if r:
                    return r
            elif b not in ("builtins.object", "typing.Generic", "typing.Protocol"):
                return b
        return None

    def class_attr_type(self, cls_id: str, attr: str, seen: Optional[set] = None) -> Optional[str]:
        seen = set() if seen is None else seen
        if cls_id in seen or cls_id not in self.defs:
            return None
        seen.add(cls_id)
        types = self.class_attr_types.get(cls_id, {}).get(attr)
        if types and len(types) == 1:
            return next(iter(types))
        for b in self.defs[cls_id].bases:
            r = self.class_attr_type(b, attr, seen)
            if r:
                return r
        return None

    def global_var_type(self, fq: str, depth: int = 0) -> Optional[str]:
        mod, _, name = fq.rpartition(".")
        mi = self.modules.get(mod)
        if mi is None or depth > 8:
            return None
        t = self.module_var_types.get(mod, {}).get(name)
        if t:
            return t
        if name in mi.imports:
            return self.global_var_type(mi.imports[name], depth + 1)
        return None

    # ---- type inference ----------------------------------------------------

    def find_alias(self, fq: str) -> Optional[tuple]:
        mod, _, name = fq.rpartition(".")
        mi = self.modules.get(mod)
        if mi is None:
            return None
        if name in mi.aliases:
            return mi, mi.aliases[name]
        if name in mi.imports:
            return self.find_alias(mi.imports[name])
        return None

    def ann_to_type(self, expr: Optional[ast.AST], mi: ModuleInfo, depth: int = 0) -> Optional[str]:
        """Annotation -> class id / external dotted name (unions of one type only)."""
        if expr is None or depth > 10:
            return None
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, str):
                try:
                    return self.ann_to_type(ast.parse(expr.value, mode="eval").body, mi, depth + 1)
                except SyntaxError:
                    return None
            return None
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
            return self._union([expr.left, expr.right], mi, depth)
        if isinstance(expr, ast.Subscript):
            base = dotted(expr.value) or []
            last = base[-1] if base else ""
            sl = expr.slice
            elts = list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]
            if last == "Optional":
                return self._union(elts, mi, depth)
            if last == "Union":
                return self._union(elts, mi, depth)
            if last == "Annotated":
                return self.ann_to_type(elts[0], mi, depth + 1)
            if last in ("Type", "ClassVar", "Final"):
                return None if last == "Type" else self.ann_to_type(elts[0], mi, depth + 1)
            return self.ann_to_type(expr.value, mi, depth + 1)
        parts = dotted(expr)
        if not parts:
            return None
        if parts == ["None"]:
            return None
        fq = self.module_expand(mi, parts)
        if fq is None:
            return None
        target = self.resolve_fq(fq)
        if target:
            return target if self.defs[target].kind == "class" else None
        alias = self.find_alias(fq)
        if alias:
            ami, aexpr = alias
            if isinstance(aexpr, ast.Call):
                fparts = dotted(aexpr.func) or []
                if fparts and fparts[-1] == "TypeVar":
                    for kw in aexpr.keywords:
                        if kw.arg == "bound":
                            return self.ann_to_type(kw.value, ami, depth + 1)
                return None
            return self.ann_to_type(aexpr, ami, depth + 1)
        if self.in_project(fq):
            return None
        return fq

    def _union(self, members: list, mi: ModuleInfo, depth: int) -> Optional[str]:
        types = set()
        for m in members:
            if isinstance(m, ast.Constant) and m.value is None:
                continue
            if dotted(m) == ["None"]:
                continue
            t = self.ann_to_type(m, mi, depth + 1)
            if t is None:
                return None
            types.add(t)
        return next(iter(types)) if len(types) == 1 else None

    def return_type(self, def_id: str) -> Optional[str]:
        if def_id in self._ret_cache:
            return self._ret_cache[def_id]
        self._ret_cache[def_id] = None  # recursion guard
        d = self.defs[def_id]
        t = None
        if isinstance(d.node, FUNC_TYPES):
            t = self.ann_to_type(d.node.returns, self.modules[d.module])
        self._ret_cache[def_id] = t
        return t

    def is_property(self, def_id: str) -> bool:
        return any(x in PROPERTY_DECORATORS for x in self.defs[def_id].decorators)

    def instance_type(self, expr: ast.AST, ctx: Ctx, depth: int = 0) -> Optional[str]:
        """Best-effort static type of an expression's value."""
        if depth > 8:
            return None
        lit = literal_type(expr)
        if lit:
            return lit
        if isinstance(expr, ast.Name):
            t = ctx.var_type(expr.id)
            if t or ctx.is_local(expr.id):
                return t
            mi = ctx.mi
            if expr.id in self.module_var_types.get(mi.name, {}):
                return self.module_var_types[mi.name][expr.id]
            if expr.id in mi.imports:
                return self.global_var_type(mi.imports[expr.id])
            return None
        if isinstance(expr, ast.Attribute):
            if is_super_call(expr.value):
                return None
            base = self.instance_type(expr.value, ctx, depth + 1)
            if base and base in self.defs and self.defs[base].kind == "class":
                at = self.class_attr_type(base, expr.attr)
                if at:
                    return at
                m = self.lookup_member(base, expr.attr)
                if m and self.is_property(m):
                    return self.return_type(m)
                return None
            parts = dotted(expr)
            if parts and not ctx.is_local(parts[0]):
                fq = self.module_expand(ctx.mi, parts)
                if fq:
                    return self.global_var_type(fq)
            return None
        if isinstance(expr, ast.Await):
            return self.instance_type(expr.value, ctx, depth + 1)
        if isinstance(expr, ast.Call):
            for target, kind, _how in self.resolve_call(expr, ctx, depth + 1):
                if kind == "instantiate":
                    return target
                if target in self.defs:
                    return self.return_type(target)
                if target.startswith("?"):
                    return None
                if target in KNOWN_RETURNS:
                    return KNOWN_RETURNS[target]
                last = target.rsplit(".", 1)[-1]
                if last[:1].isupper():  # external Class(...) heuristic
                    return target
                return None
        return None

    # ---- contexts ----------------------------------------------------------

    def module_ctx(self, module: str) -> Ctx:
        key = ("module", module)
        if key not in self._ctx_cache:
            self._ctx_cache[key] = Ctx(d=self.defs[module], mi=self.modules[module])
        return self._ctx_cache[key]

    def ctx_for(self, def_id: str) -> Ctx:
        d = self.defs[def_id]
        if d.kind == "module":
            return self.module_ctx(def_id)
        if def_id in self._ctx_cache:
            return self._ctx_cache[def_id]
        mi = self.modules[d.module]
        pdef = self.defs[d.parent] if d.parent else None
        parent_ctx = None
        if pdef is not None and pdef.kind in ("function", "method"):
            parent_ctx = self.ctx_for(pdef.id)
        chain = list(parent_ctx.func_chain) if parent_ctx else []
        if d.kind == "class":
            ctx = Ctx(d=d, mi=mi, parent=parent_ctx, func_chain=chain)
            self._ctx_cache[def_id] = ctx
            return ctx
        cls = d.parent if d.kind == "method" else (parent_ctx.cls if parent_ctx else None)
        ctx = Ctx(d=d, mi=mi, cls=cls, parent=parent_ctx, func_chain=[d, *chain])
        self._ctx_cache[def_id] = ctx  # cache early: guards recursion

        node = d.node
        args = node.args  # type: ignore[attr-defined]
        positional = list(args.posonlyargs) + list(args.args)
        all_args = positional + list(args.kwonlyargs)
        for a in (args.vararg, args.kwarg):
            if a is not None:
                ctx.local_names.add(a.arg)
        types: dict = defaultdict(set)
        for i, a in enumerate(all_args):
            ctx.local_names.add(a.arg)
            if i == 0 and d.kind == "method" and a in positional:
                if "staticmethod" in d.decorators:
                    pass
                elif "classmethod" in d.decorators:
                    ctx.local_classes[a.arg] = d.parent
                    continue
                else:
                    types[a.arg].add(d.parent)
                    continue
            t = self.ann_to_type(a.annotation, mi)
            if t:
                types[a.arg].add(t)

        body_nodes = [node, *d.extra_nodes]
        for bn in body_nodes:
            for n in walk_scope(bn):
                if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                    for t in targets:
                        if isinstance(t, (ast.Name, ast.Tuple, ast.List, ast.Starred)):
                            ctx.local_names.update(bound_names(t))
                elif isinstance(n, (ast.For, ast.AsyncFor, ast.comprehension)):
                    ctx.local_names.update(bound_names(n.target))
                elif isinstance(n, ast.withitem) and n.optional_vars is not None:
                    ctx.local_names.update(bound_names(n.optional_vars))
                elif isinstance(n, ast.ExceptHandler) and n.name:
                    ctx.local_names.add(n.name)
                elif isinstance(n, ast.NamedExpr):
                    ctx.local_names.add(n.target.id)
                # function-local imports are not "unknown locals": they are
                # already in the module import table (collected module-wide).
        ctx.local_names -= set(d.local_defs)

        def publish():
            ctx.local_types = {k: next(iter(v)) for k, v in types.items() if len(v) == 1}

        publish()
        for bn in body_nodes:
            for n in walk_scope(bn):
                t = None
                name = None
                if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                    name, t = n.targets[0].id, self.instance_type(n.value, ctx)
                elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                    name, t = n.target.id, self.ann_to_type(n.annotation, mi)
                elif isinstance(n, ast.withitem) and isinstance(n.optional_vars, ast.Name):
                    name, t = n.optional_vars.id, self.instance_type(n.context_expr, ctx)
                if name and t:
                    types[name].add(t)
                    publish()
        return ctx

    # ---- inference passes --------------------------------------------------

    def infer_types(self, rounds: int = 2) -> None:
        for _ in range(rounds):
            self._ctx_cache.clear()
            self._ret_cache.clear()
            self._infer_module_vars()
            attr_types: dict = defaultdict(lambda: defaultdict(set))
            for d in list(self.defs.values()):
                if d.kind == "class":
                    for n in d.node.body:  # type: ignore[attr-defined]
                        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                            t = self.ann_to_type(n.annotation, self.modules[d.module])
                            if t:
                                attr_types[d.id][n.target.id].add(t)
                elif d.kind == "method":
                    args = d.node.args  # type: ignore[attr-defined]
                    params = list(args.posonlyargs) + list(args.args)
                    if not params or "staticmethod" in d.decorators or "classmethod" in d.decorators:
                        continue
                    self_name = params[0].arg
                    ctx = self.ctx_for(d.id)
                    for bn in [d.node, *d.extra_nodes]:
                        for n in walk_scope(bn):
                            tgt, t = None, None
                            if isinstance(n, ast.Assign) and len(n.targets) == 1:
                                tgt = n.targets[0]
                                val = n.value
                                if self._is_self_attr(tgt, self_name):
                                    t = self.instance_type(val, ctx)
                            elif isinstance(n, ast.AnnAssign):
                                tgt = n.target
                                if self._is_self_attr(tgt, self_name):
                                    t = self.ann_to_type(n.annotation, ctx.mi)
                            if t and tgt is not None:
                                attr_types[d.parent][tgt.attr].add(t)  # type: ignore[attr-defined]
            self.class_attr_types = {k: dict(v) for k, v in attr_types.items()}
        self._ctx_cache.clear()
        self._ret_cache.clear()

    @staticmethod
    def _is_self_attr(t: ast.AST, self_name: str) -> bool:
        return isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == self_name

    def _infer_module_vars(self) -> None:
        for mi in self.modules.values():
            ctx = self.module_ctx(mi.name)
            out = self.module_var_types.setdefault(mi.name, {})
            for n in walk_scope(mi.tree):
                if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                    t = self.instance_type(n.value, ctx)
                    if t:
                        out[n.targets[0].id] = t
                elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                    t = self.ann_to_type(n.annotation, mi)
                    if t:
                        out[n.target.id] = t

    # ---- call resolution ---------------------------------------------------

    def resolve_call(self, call: ast.Call, ctx: Ctx, depth: int = 0) -> list:
        """Return [(target, kind, how)] for one ast.Call."""
        f = call.func
        if depth > 8:
            return [("?", "call", "unresolved")]
        if isinstance(f, ast.Name):
            lc = ctx.local_class(f.id)
            if lc:
                return self._finish((lc, "self"))
            if ctx.is_local(f.id) and not any(f.id in fd.local_defs for fd in ctx.func_chain):
                t = ctx.var_type(f.id)
                if t and t in self.defs and self.defs[t].kind == "class":
                    m = self.lookup_member(t, "__call__")
                    if m:
                        return [(m, "call", "type")]
                return [(f"?{f.id}", "call", "unresolved")]
            sym = self.resolve_name(ctx, f.id)
            return self._finish(sym, f.id)
        if isinstance(f, ast.Attribute):
            if is_super_call(f.value) and ctx.cls:
                for b in self.defs[ctx.cls].bases:
                    if b in self.defs:
                        m = self.lookup_member(b, f.attr)
                        if m:
                            return [(m, "call", "super")]
                    else:
                        return [(f"{b}.{f.attr}", "call", self._ext_how(b))]
                ext = self.external_base(ctx.cls)
                if ext:
                    return [(f"{ext}.{f.attr}", "call", self._ext_how(ext))]
                return [(f"?super.{f.attr}", "call", "unresolved")]
            parts = dotted(f)
            lc = ctx.local_class(parts[0]) if parts else None
            if lc and len(parts) == 2:  # cls.method(...)
                m = self.lookup_member(lc, f.attr)
                if m:
                    return [(m, "call", "self")]
            recv_type = self.instance_type(f.value, ctx, depth + 1)
            if recv_type:
                how = "self" if isinstance(f.value, ast.Name) and recv_type == ctx.cls else "type"
                if recv_type in self.defs and self.defs[recv_type].kind == "class":
                    m = self.lookup_member(recv_type, f.attr)
                    if m:
                        return [(m, "call", how)]
                    ext = self.external_base(recv_type)
                    if ext:
                        return [(f"{ext}.{f.attr}", "call", self._ext_how(ext))]
                    return [(f"?{recv_type}.{f.attr}", "call", "unresolved")]
                if recv_type not in self.defs:
                    return [(f"{recv_type}.{f.attr}", "call", self._ext_how(recv_type))]
            if parts and not ctx.is_local(parts[0]):
                sym = self.resolve_name(ctx, parts[0], parts[1:])
                return self._finish(sym, ".".join(parts))
            return [(f"?.{f.attr}", "call", "unresolved")]
        return [("?", "call", "unresolved")]

    @staticmethod
    def _ext_how(name: str) -> str:
        return "builtin" if name.startswith("builtins.") else "external"

    def _finish(self, sym: Optional[tuple], label: str = "") -> list:
        if sym is None:
            return [(f"?{label}", "call", "unresolved")]
        target, how = sym
        d = self.defs.get(target)
        if d is None:
            return [(target, "call", how)]
        if d.kind == "class":
            out = [(target, "instantiate", how)]
            init = self.lookup_member(target, "__init__")
            if init:
                out.append((init, "call", "constructor"))
            return out
        if d.kind == "module":
            return [(f"?{target}", "call", "unresolved")]
        return [(target, "call", how)]

    # ---- graph -------------------------------------------------------------

    def source_spans(self, d: Def) -> list:
        """[(first, last), ...] 1-based line spans, decorators included.

        One span per definition: a name redefined in the same scope (property
        setter, @overload, if/else branches) shares one id but keeps each
        definition's own span, so code between them is not included.
        """
        if d.kind == "module":
            return [(1, len(self.sources[d.module]))] if self.sources[d.module] else []
        spans = []
        for n in [d.node, *d.extra_nodes]:
            first = min([n.lineno, *(x.lineno for x in getattr(n, "decorator_list", []))])
            spans.append((first, n.end_lineno or n.lineno))
        return spans

    def source_text(self, d: Def) -> str:
        """Verbatim source of each span; spans joined by one blank line."""
        lines = self.sources[d.module]
        parts = []
        for first, last in self.source_spans(d):
            chunk = "".join(lines[first - 1 : last])
            parts.append(chunk if chunk.endswith("\n") else chunk + "\n")
        return "\n".join(parts)

    def build(
        self,
        include_external: bool,
        include_builtins: bool,
        include_unresolved: bool,
        include_text: bool = True,
        edges_key: str = "edges",
    ) -> dict:
        """Return networkx node_link data: directed multigraph, edges keyed per (u, v)."""
        self.infer_types()
        agg: dict = {}
        stats = Counter()
        for did in sorted(self.defs):
            d = self.defs[did]
            ctx = self.ctx_for(did)
            for node in [d.node, *d.extra_nodes]:
                for n in walk_scope(node):
                    if not isinstance(n, ast.Call):
                        continue
                    stats["call_sites"] += 1
                    results = self.resolve_call(n, ctx)
                    primary = self._category(results[0][0])
                    stats[f"call_sites_{primary}"] += 1
                    for target, kind, how in results:
                        cat = self._category(target)
                        if cat == "external" and not include_external:
                            continue
                        if cat == "builtin" and not include_builtins:
                            continue
                        if cat == "unresolved" and not include_unresolved:
                            continue
                        key = (did, target, kind)
                        e = agg.setdefault(key, {"resolution": how, "lines": set()})
                        e["lines"].add(n.lineno)

        nodes = []
        for did in sorted(self.defs):
            d = self.defs[did]
            node = {
                "id": did,
                "kind": d.kind,
                "name": d.name,
                "qualname": did[len(d.module) + 1 :] if d.kind != "module" else "",
                "module": d.module,
                "file": d.file,
                "lineno": d.lineno,
                "end_lineno": d.end_lineno,
                "parent": d.parent,
            }
            if d.kind in ("function", "method"):
                node["async"] = d.is_async
            if d.decorators:
                node["decorators"] = d.decorators
            if d.kind == "class":
                node["bases"] = d.bases
            if include_text:
                spans = self.source_spans(d)
                node["text_lineno"] = spans[0][0] if spans else 1
                node["text_spans"] = [list(s) for s in spans]
                node["text"] = self.source_text(d)
            nodes.append(node)
            stats[f"nodes_{d.kind}"] += 1

        extra = set()
        edges = []
        next_key: Counter = Counter()  # multigraph key: 0, 1, ... per (source, target)
        for (src, tgt, kind), e in sorted(agg.items()):
            if tgt not in self.defs:
                extra.add(tgt)
            lines = sorted(e["lines"])
            edges.append(
                {
                    "source": src,
                    "target": tgt,
                    "key": next_key[(src, tgt)],
                    "type": EDGE_TYPES[kind],
                    "resolution": e["resolution"],
                    "count": len(lines),
                    "lines": lines,
                }
            )
            next_key[(src, tgt)] += 1
        for t in sorted(extra):
            ext = {"id": t, "kind": self._category(t), "name": t.lstrip("?").rsplit(".", 1)[-1]}
            if include_text:
                ext["text"] = ""  # no source available outside the analysed tree
            nodes.append(ext)

        resolved = stats["call_sites_internal"] + stats["call_sites_external"] + stats["call_sites_builtin"]
        total = stats["call_sites"] or 1
        summary = {
            "modules": stats["nodes_module"],
            "classes": stats["nodes_class"],
            "functions": stats["nodes_function"],
            "methods": stats["nodes_method"],
            "call_sites": stats["call_sites"],
            "call_sites_internal": stats["call_sites_internal"],
            "call_sites_external": stats["call_sites_external"],
            "call_sites_builtin": stats["call_sites_builtin"],
            "call_sites_unresolved": stats["call_sites_unresolved"],
            "resolved_ratio": round(resolved / total, 4),
            "edges": len(edges),
            "internal_edges": sum(1 for e in edges if e["target"] in self.defs),
        }
        return {
            "directed": True,
            "multigraph": True,
            "graph": {
                "generator": f"tools/callgraph/callgraph.py {__version__}",
                "schema": 3,
                "format": f"networkx node_link (edges under '{edges_key}')",
                "root": self.root.resolve().name,
                "commit": git_commit(self.root),
                "python": sys.version.split()[0],
                "options": {
                    "include_external": include_external,
                    "include_builtins": include_builtins,
                    "include_unresolved": include_unresolved,
                    "include_text": include_text,
                },
                "warnings": self.warnings,
                "stats": summary,
            },
            "nodes": nodes,
            edges_key: edges,
        }

    def _category(self, target: str) -> str:
        if target in self.defs:
            return "internal"
        if target.startswith("?"):
            return "unresolved"
        if target.startswith("builtins."):
            return "builtin"
        return "external"


def git_commit(root: Path) -> Optional[str]:
    """Commit of the analysed checkout (git first: on pull_request events
    GITHUB_SHA is a synthetic merge commit, not a commit in the branch)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        if out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return os.environ.get("GITHUB_SHA") or None


def graph_edges(graph: dict) -> list:
    return graph["edges"] if "edges" in graph else graph["links"]


def markdown_summary(graph: dict, top: int = 15) -> str:
    s = graph["graph"]["stats"]
    kinds = {n["id"]: n["kind"] for n in graph["nodes"]}
    internal = [e for e in graph_edges(graph) if kinds.get(e["target"]) in ("function", "method", "class")]
    fan_in = Counter()
    fan_out = Counter()
    for e in internal:
        fan_in[e["target"]] += e["count"]
        fan_out[e["source"]] += e["count"]
    lines = [
        "## Call graph",
        "",
        f"Commit `{(graph['graph']['commit'] or 'unknown')[:12]}`",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for k in (
        "modules",
        "classes",
        "functions",
        "methods",
        "call_sites",
        "call_sites_internal",
        "call_sites_external",
        "call_sites_builtin",
        "call_sites_unresolved",
        "resolved_ratio",
        "edges",
        "internal_edges",
    ):
        lines.append(f"| {k} | {s[k]} |")
    lines += ["", f"### Most called (top {top})", "", "| callee | calls |", "|---|---|"]
    lines += [f"| `{k}` | {v} |" for k, v in fan_in.most_common(top)]
    lines += ["", f"### Largest fan-out (top {top})", "", "| caller | calls |", "|---|---|"]
    lines += [f"| `{k}` | {v} |" for k, v in fan_out.most_common(top)]
    return "\n".join(lines) + "\n"


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("paths", nargs="+", help="package dirs or .py files to analyse")
    p.add_argument("-o", "--output", default="graph.json")
    p.add_argument("--root", default=".", help="import root used to derive module names (default: .)")
    p.add_argument("--exclude", action="append", default=[], help="directory name to skip (repeatable)")
    p.add_argument("--no-external", action="store_true", help="drop edges to third-party/stdlib targets")
    p.add_argument("--include-builtins", action="store_true", help="keep edges to builtins (len, isinstance, ...)")
    p.add_argument("--include-unresolved", action="store_true", help="keep edges to unresolved targets ('?name')")
    p.add_argument("--no-text", action="store_true", help="omit each node's source text")
    p.add_argument(
        "--edges-key",
        choices=("edges", "links"),
        default="edges",
        help="top-level key holding the edge list (networkx >=3.4 'edges'; older default 'links')",
    )
    p.add_argument("--summary", help="also write a markdown summary to this path")
    a = p.parse_args(argv)

    root = Path(a.root)
    files = list(iter_py_files(a.paths, DEFAULT_EXCLUDES | set(a.exclude)))
    if not files:
        print("no .py files found", file=sys.stderr)
        return 2
    proj = Project(root)
    proj.load(files)
    graph = proj.build(
        include_external=not a.no_external,
        include_builtins=a.include_builtins,
        include_unresolved=a.include_unresolved,
        include_text=not a.no_text,
        edges_key=a.edges_key,
    )
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if a.summary:
        Path(a.summary).write_text(markdown_summary(graph), encoding="utf-8")
    s = graph["graph"]["stats"]
    print(
        f"{len(files)} files | {s['classes']} classes, {s['functions']} functions, {s['methods']} methods | "
        f"{s['call_sites']} call sites, {s['resolved_ratio']:.1%} resolved | {s['edges']} edges -> {out}"
    )
    for w in proj.warnings:
        print(f"warning: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
