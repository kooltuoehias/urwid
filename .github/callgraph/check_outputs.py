#!/usr/bin/env python3
# Standalone CI script (call graph), not urwid library code.
# ruff: noqa: D103, D403, EXE001
"""Validate graph.json (and optionally the embeddings npz) produced in CI.

python tools/callgraph/check_outputs.py out/graph.json
python tools/callgraph/check_outputs.py out/graph.json --embeddings out/embeddings
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import textwrap
from pathlib import Path

__all__ = (
    "check_embeddings",
    "check_graph",
    "check_node_link",
    "graph_edges",
    "main",
    "parses_as_def",
    "text_segments",
)


TEXT_KINDS = ("module", "class", "function", "method")
DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def text_segments(n: dict) -> list:
    """Split a node's text back into its per-definition spans (see text_spans)."""
    lines = n["text"].splitlines(keepends=True)
    spans = n.get("text_spans") or [[1, len(lines)]]
    out, pos = [], 0
    for i, (first, last) in enumerate(spans):
        size = last - first + 1
        out.append("".join(lines[pos : pos + size]))
        pos += size + (1 if i < len(spans) - 1 else 0)  # blank separator line
    if pos != len(lines):
        return []
    return out


def parses_as_def(n: dict) -> bool:
    """Each span of the text must parse to exactly one def/class with the node's name."""
    segments = text_segments(n)
    if not segments:
        return False
    for seg in segments:
        try:
            body = ast.parse(textwrap.dedent(seg)).body
        except SyntaxError:
            return False
        if len(body) != 1 or not isinstance(body[0], DEF_TYPES) or body[0].name != n["name"]:
            return False
    return True


def graph_edges(g: dict) -> list:
    return g["edges"] if "edges" in g else g["links"]


def check_node_link(g: dict) -> list:
    """networkx node_link / swegemma envelope and edge fields."""
    errs = []
    if g.get("directed") is not True or g.get("multigraph") is not True:
        errs.append("top level must have directed=true, multigraph=true")
    if not isinstance(g.get("graph"), dict):
        errs.append("top level must have a 'graph' object")
    if ("edges" in g) == ("links" in g):
        errs.append("exactly one of 'edges' / 'links' must be present")
        return errs
    seen = set()
    for e in graph_edges(g):
        for f in ("source", "target", "key", "type"):
            if f not in e:
                errs.append(f"edge missing '{f}': {e}")
                return errs
        if e["type"] not in ("calls", "instantiates"):
            errs.append(f"unknown edge type {e['type']!r}")
        k = (e["source"], e["target"], e["key"])
        if k in seen:
            errs.append(f"duplicate (source, target, key): {k}")
        seen.add(k)
    for n in g["nodes"]:
        if not ({"id", "name", "text"} <= n.keys()) and g["graph"]["options"].get("include_text", False):
            errs.append(f"node missing id/name/text: {n.get('id')}")
            break
    return errs


def check_graph(g: dict) -> list:
    errs = check_node_link(g)
    if errs:
        return errs
    s = g["graph"]["stats"]
    if not (s["classes"] and s["methods"] and s["functions"]):
        errs.append(f"empty graph stats: {s}")
    if s["internal_edges"] <= 0:
        errs.append("no internal edges")
    ids = [n["id"] for n in g["nodes"]]
    if len(ids) != len(set(ids)):
        errs.append("duplicate node ids")
    idset = set(ids)
    dangling = [e for e in graph_edges(g) if e["source"] not in idset or e["target"] not in idset]
    if dangling:
        errs.append(f"{len(dangling)} dangling edges, e.g. {dangling[:3]}")
    if g["graph"]["options"].get("include_text", False):
        for n in g["nodes"]:
            if "text" not in n:
                errs.append(f"node without text field: {n['id']}")
            elif n["kind"] in ("class", "function", "method") and not n["text"].strip():
                errs.append(f"empty text: {n['id']}")  # empty modules (blank __init__.py) are fine
            elif n["kind"] in ("class", "function", "method") and not parses_as_def(n):
                errs.append(f"text of {n['id']} is not exactly its definition")
    return errs


def check_embeddings(g: dict, emb_dir: Path) -> list:
    import numpy as np

    errs = []
    sidecars = sorted(emb_dir.glob("*.json"))
    if len(sidecars) != 1:
        return [f"expected one sidecar json in {emb_dir}, found {len(sidecars)}"]
    meta = json.loads(sidecars[0].read_text())
    npz_path = emb_dir / meta["file"]
    expected_name = f"{meta['repo']}_{g['graph']['commit']}.npz"
    if npz_path.name != expected_name:
        errs.append(f"npz name {npz_path.name} != {expected_name}")
    data = np.load(npz_path)
    dim = meta["dim"]
    text_ids = {n["id"] for n in g["nodes"] if n["kind"] in TEXT_KINDS and n.get("text", "").strip()}
    keys = set(data.files)
    if keys != text_ids:
        errs.append(f"npz keys != text nodes: missing {len(text_ids - keys)}, extra {len(keys - text_ids)}")
    ordered = sorted(keys & text_ids)
    mat = np.stack([data[k] for k in ordered])
    if mat.dtype != np.float32 or mat.shape[1] != dim:
        errs.append(f"bad dtype/shape {mat.dtype} {mat.shape}")
    if not np.isfinite(mat).all():
        errs.append("non-finite values")
    norms = np.linalg.norm(mat, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        errs.append(f"vectors not normalised (min {norms.min():.4f}, max {norms.max():.4f})")
    if len(np.unique(mat.round(5), axis=0)) < 0.9 * len(mat):
        errs.append("too many identical vectors")

    # Semantic smoke test: methods sharing a name across classes (Client.get /
    # AsyncClient.get, ...) should be much closer than random pairs.
    if not meta.get("backend", "").startswith("hash"):
        name_of = {n["id"]: n["name"] for n in g["nodes"]}
        idx = {k: i for i, k in enumerate(ordered)}
        by_name: dict = {}
        for k in ordered:
            if not name_of[k].startswith("__"):
                by_name.setdefault(name_of[k], []).append(idx[k])
        pairs = [(v[0], v[1]) for v in by_name.values() if len(v) >= 2]
        rng = np.random.default_rng(0)
        rand = rng.integers(0, len(ordered), size=(2000, 2))
        rand = rand[rand[:, 0] != rand[:, 1]]
        same = float(np.mean([mat[i] @ mat[j] for i, j in pairs])) if pairs else 0.0
        base = float(np.mean(np.sum(mat[rand[:, 0]] * mat[rand[:, 1]], axis=1)))
        print(f"semantic smoke: same-name pairs cos={same:.3f} (n={len(pairs)}), random pairs cos={base:.3f}")
        if not pairs or same < base + 0.05:
            errs.append("embeddings show no semantic structure (same-name pairs not closer than random)")
    print(f"embeddings: {len(keys)} vectors, dim {dim}, model {meta['model']}")
    return errs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("graph")
    p.add_argument("--embeddings", help="directory holding <repo>_<commit>.npz + .json")
    a = p.parse_args()
    g = json.loads(Path(a.graph).read_text(encoding="utf-8"))
    errs = check_graph(g)
    if a.embeddings:
        errs += check_embeddings(g, Path(a.embeddings))
    if errs:
        print("\n".join(f"ERROR: {e}" for e in errs))
        return 1
    print(f"OK: {len(g['nodes'])} nodes, {len(graph_edges(g))} edges, stats={g['graph']['stats']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
