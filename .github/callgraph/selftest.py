#!/usr/bin/env python3
# Standalone CI script (call graph), not urwid library code.
# ruff: noqa: D103, EXE001, PERF401, RUF100, S101
"""Self-test for callgraph.py on a small fixture package (no pytest needed).

Named selftest.py (not test_*.py) so httpx's own pytest run won't collect it.
"""

from __future__ import annotations

import json
import sys
import tempfile
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import callgraph  # noqa: E402
import check_outputs  # noqa: E402

FILES = {
    "pkg/__init__.py": """
        from .core import Engine, helper
        from . import util
    """,
    "pkg/util.py": """
        import os

        def join_path(a, b):
            return os.path.join(a, b)

        def log(msg):
            print(msg)
    """,
    "pkg/core.py": """
        import typing
        from .util import log, join_path
        from . import util

        T = typing.TypeVar("T", bound="Engine")

        def helper(x):
            log(x)
            return util.join_path(x, "y")

        class Part:
            def spin(self) -> None:
                log("spin")

            def name(self) -> str:
                return "p"

        class Base:
            def __init__(self) -> None:
                self.part = Part()

            def start(self) -> "Part":
                self.prepare()
                return self.part

            def prepare(self) -> None:
                pass

            @property
            def main_part(self) -> Part:
                return self.part

            @classmethod
            def build(cls):
                return cls()

        class Engine(Base):
            def __init__(self, fuel: int) -> None:
                super().__init__()
                self.fuel = fuel

            def __enter__(self: T) -> T:
                return self

            def __exit__(self, *args):
                pass

            def prepare(self) -> None:
                super().prepare()
                helper("x")

            def run(self, other: typing.Optional["Part"] = None) -> None:
                self.start().spin()
                self.part.spin()
                self.main_part.name()
                if other is not None:
                    other.name()

                def inner():
                    self.prepare()

                inner()
                p = Part()
                p.spin()
                len([1])
                unknown_obj().foo()

        def make() -> Engine:
            return Engine(1)

        def use():
            e = make()
            e.run()
            with Engine(2) as eng:
                eng.prepare()
            make().start()
            Engine.prepare(e)

        helper("module-level")
    """,
    "pkg/empty/__init__.py": "",
    "pkg/branchy.py": """
        def make_handler(flag):
            if flag:

                def handler(x):
                    return x + 1

            elif flag is None:
                marker = "BETWEEN"

                def handler(x):
                    return x - 1

            else:

                def handler(x):
                    return x

            return handler
    """,
    "pkg/cli.py": """
        import pkg
        from pkg import Engine as E

        async def fetch() -> "E":
            return E(0)

        def main():
            pkg.Engine(3).run()
            E(4)
            pkg.helper("z")

        async def amain():
            eng = await fetch()
            eng.prepare()
    """,
}

C = "pkg.core"
EXPECTED = [
    # (source, target, kind)
    (f"{C}.helper", "pkg.util.log", "call"),
    (f"{C}.helper", "pkg.util.join_path", "call"),
    ("pkg.util.join_path", "os.path.join", "call"),
    ("pkg.util.log", "builtins.print", "call"),
    (f"{C}.Part.spin", "pkg.util.log", "call"),
    (f"{C}.Base.__init__", f"{C}.Part", "instantiate"),
    (f"{C}.Base.start", f"{C}.Base.prepare", "call"),
    (f"{C}.Base.build", f"{C}.Base", "instantiate"),
    (f"{C}.Engine.__init__", f"{C}.Base.__init__", "call"),
    (f"{C}.Engine.prepare", f"{C}.Base.prepare", "call"),
    (f"{C}.Engine.prepare", f"{C}.helper", "call"),
    (f"{C}.Engine.run", f"{C}.Base.start", "call"),
    (f"{C}.Engine.run", f"{C}.Part.spin", "call"),
    (f"{C}.Engine.run", f"{C}.Part.name", "call"),
    (f"{C}.Engine.run", f"{C}.Engine.run.<locals>.inner", "call"),
    (f"{C}.Engine.run", f"{C}.Part", "instantiate"),
    (f"{C}.Engine.run", "builtins.len", "call"),
    (f"{C}.Engine.run.<locals>.inner", f"{C}.Engine.prepare", "call"),
    (f"{C}.make", f"{C}.Engine", "instantiate"),
    (f"{C}.make", f"{C}.Engine.__init__", "call"),
    (f"{C}.use", f"{C}.make", "call"),
    (f"{C}.use", f"{C}.Engine.run", "call"),
    (f"{C}.use", f"{C}.Engine.prepare", "call"),
    (f"{C}.use", f"{C}.Base.start", "call"),
    (C, f"{C}.helper", "call"),
    ("pkg.cli.main", f"{C}.Engine", "instantiate"),
    ("pkg.cli.main", f"{C}.Engine.run", "call"),
    ("pkg.cli.main", f"{C}.helper", "call"),
    ("pkg.cli.amain", "pkg.cli.fetch", "call"),
    ("pkg.cli.amain", f"{C}.Engine.prepare", "call"),
]

FORBIDDEN = [
    # Base.start's self.prepare() must not be bound to the subclass override
    (f"{C}.Base.start", f"{C}.Engine.prepare", "call"),
]

EXPECTED_NODES = {
    "pkg": "module",
    f"{C}.Engine": "class",
    f"{C}.Engine.run": "method",
    f"{C}.Engine.run.<locals>.inner": "function",
    f"{C}.helper": "function",
    "pkg.cli.fetch": "function",
}


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for rel, src in FILES.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(textwrap.dedent(src).lstrip(), encoding="utf-8")
        out = root / "graph.json"
        rc = callgraph.main(
            [str(root / "pkg"), "--root", str(root), "-o", str(out), "--include-builtins", "--include-unresolved"]
        )
        assert rc == 0, rc
        g = json.loads(out.read_text())

    edge_list = check_outputs.graph_edges(g)
    kind_of = {"calls": "call", "instantiates": "instantiate"}
    edges = {(e["source"], e["target"], kind_of[e["type"]]) for e in edge_list}
    failures = list(check_outputs.check_node_link(g))
    kinds = {n["id"]: n["kind"] for n in g["nodes"]}
    for exp in EXPECTED:
        if exp not in edges:
            failures.append(f"missing edge   {exp}")
    for bad in FORBIDDEN:
        if bad in edges:
            failures.append(f"unexpected edge {bad}")
    for nid, kind in EXPECTED_NODES.items():
        if kinds.get(nid) != kind:
            failures.append(f"node {nid}: expected {kind}, got {kinds.get(nid)}")
    if not any(e["target"].startswith("?") for e in edge_list):
        failures.append("expected the unknown_obj() call to be reported as unresolved")
    for e in edge_list:
        assert e["count"] == len(e["lines"]) > 0, e

    # node source text
    by_id = {n["id"]: n for n in g["nodes"]}
    util_src = textwrap.dedent(FILES["pkg/util.py"]).lstrip()
    text_checks = [
        ("pkg.util", lambda t: t == util_src, "module text is the whole file"),
        (f"{C}.helper", lambda t: t.startswith("def helper(x):") and t.rstrip().endswith('"y")'), "function span"),
        (f"{C}.Base.main_part", lambda t: t.startswith("    @property\n    def main_part"), "decorators included"),
        (f"{C}.Engine.run.<locals>.inner", lambda t: t.strip() == "def inner():\n            self.prepare()", "nested"),
        (f"{C}.Part", lambda t: "def spin" in t and "def name" in t and "class Base" not in t, "class span"),
        ("os.path.join", lambda t: t == "", "external has empty text"),
        ("pkg.empty", lambda t: t == "", "empty __init__ has empty text"),
        (
            "pkg.branchy.make_handler.<locals>.handler",
            lambda t: t.count("def handler") == 3 and "BETWEEN" not in t and "elif" not in t,
            "branch redefinitions: each def only, no code between",
        ),
    ]
    for nid, ok, what in text_checks:
        t = by_id.get(nid, {}).get("text")
        if t is None or not ok(t):
            failures.append(f"text check failed ({what}): {nid} -> {t!r}")
    for n in g["nodes"]:
        if n["kind"] in ("class", "function", "method") and not n.get("text"):
            failures.append(f"empty text on {n['id']}")
        if n["kind"] in ("class", "function", "method") and not check_outputs.parses_as_def(n):
            failures.append(f"text of {n['id']} is not exactly its definition")

    if failures:
        print("\n".join(failures))
        print(f"FAILED {len(failures)} check(s)")
        return 1
    print(f"OK: {len(EXPECTED)} edges, {len(EXPECTED_NODES)} nodes, stats={g['graph']['stats']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
