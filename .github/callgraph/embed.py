#!/usr/bin/env python3
# Standalone CI script (call graph), not urwid library code.
# ruff: noqa: B905, C416, D103, EXE001, UP045
"""Embed each graph node's source text -> embeddings/<repo>_<commit>.npz.

npz layout: key = node id, value = float32[dim] (default 256), L2-normalised.
Only nodes with non-empty ``text`` (module/class/function/method) are embedded;
external/builtin/unresolved nodes have no source and are skipped.

The vector is the model's output truncated to the first ``dim`` components and
re-normalised (Matryoshka truncation), so queries embedded the same way with
the same model are directly comparable. A JSON sidecar records model, dim,
query prefix and provenance.

Usage:
    pip install numpy sentence-transformers
    python tools/callgraph/embed.py out/graph.json --out-dir out/embeddings
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = (
    "encode",
    "main",
    "truncate_normalize",
)


DEFAULT_MODEL = "Snowflake/snowflake-arctic-embed-m-v1.5"
# Prefixes recommended by the default model's card: documents get none,
# search queries get QUERY_PREFIX.
QUERY_PREFIXES = {
    "Snowflake/snowflake-arctic-embed-m-v1.5": "Represent this sentence for searching relevant passages: ",
}
TEXT_KINDS = ("module", "class", "function", "method")


def encode(texts: list, model_name: str, batch_size: int) -> tuple:
    """Return (matrix[n, native_dim] float32, info dict)."""
    if model_name.startswith("hash:"):
        # Deterministic pseudo-embeddings for plumbing tests ONLY (no semantics).
        dim = int(model_name.split(":", 1)[1])
        rows = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:8], "little")
            rows.append(np.random.default_rng(seed).standard_normal(dim))
        return np.asarray(rows, dtype=np.float32), {"backend": "hash (test only, not semantic)"}

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    mat = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=False,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    info = {
        "backend": "sentence-transformers",
        "max_seq_length_tokens": int(model.max_seq_length),
        "native_dim": int(mat.shape[1]),
    }
    return mat.astype(np.float32), info


def truncate_normalize(mat: np.ndarray, dim: int) -> np.ndarray:
    if mat.shape[1] < dim:
        raise SystemExit(f"model output dim {mat.shape[1]} < requested {dim}")
    out = mat[:, :dim].astype(np.float64)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (out / norms).astype(np.float32)


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("graph", help="graph.json produced by callgraph.py (with node text)")
    p.add_argument("--out-dir", default="embeddings")
    p.add_argument("--repo", help="repo name for the file name (default: graph meta.root)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"sentence-transformers model (default: {DEFAULT_MODEL})")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    a = p.parse_args(argv)

    graph_path = Path(a.graph)
    raw = graph_path.read_bytes()
    g = json.loads(raw)
    nodes = [n for n in g["nodes"] if n.get("kind") in TEXT_KINDS and n.get("text", "").strip()]
    if not nodes:
        print("no nodes with text; build graph.json without --no-text", file=sys.stderr)
        return 2
    ids = [n["id"] for n in nodes]
    if len(set(ids)) != len(ids):
        print("duplicate node ids in graph", file=sys.stderr)
        return 2

    mat, info = encode([n["text"] for n in nodes], a.model, a.batch_size)
    vecs = truncate_normalize(mat, a.dim)

    gmeta = g["graph"]
    repo = a.repo or gmeta.get("root") or "repo"
    commit = gmeta.get("commit") or "unknown"
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{repo}_{commit}"
    npz_path = out_dir / f"{stem}.npz"
    np.savez_compressed(npz_path, **{nid: v for nid, v in zip(ids, vecs)})

    meta = {
        "file": npz_path.name,
        "format": "npz; key = graph node id, value = float32[dim], L2-normalised",
        "repo": repo,
        "commit": commit,
        "model": a.model,
        "dim": a.dim,
        "truncation": "first `dim` output components, then L2 re-normalisation",
        "input": "node.text (raw source slice, decorators included)",
        "query_prefix": QUERY_PREFIXES.get(a.model, ""),
        "document_prefix": "",
        "count": len(ids),
        "skipped_kinds": "external, builtin, unresolved (no source text)",
        "graph_sha256": hashlib.sha256(raw).hexdigest(),
        **info,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"{len(ids)} nodes -> {npz_path} ({a.model}, dim={a.dim})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
