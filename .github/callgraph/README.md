# callgraph

Static call graph for the `urwid` package, built from the Python AST (stdlib only).
Same tool as in the httpx and litestar forks; it lives under `.github/` so the package's lint config
doesn't need changing and `pyproject.toml` stays untouched.

```sh
python .github/callgraph/callgraph.py urwid -o out/graph.json --summary out/summary.md
python .github/callgraph/embed.py out/graph.json --out-dir out/embeddings   # needs sentence-transformers
python .github/callgraph/check_outputs.py out/graph.json --embeddings out/embeddings
python .github/callgraph/selftest.py        # fixture-based checks (stdlib only)
```

CI: `.github/workflows/callgraph.yml` runs the self-test, builds `graph.json`
(with node source text), embeds every node, validates both, writes a summary
to the job page and uploads `out/` as the `callgraph-<commit>` artifact:

```
out/
  graph.json
  summary.md
  embeddings/<repo>_<commit>.npz    # node id -> float32[256]
  embeddings/<repo>_<commit>.json   # model, dim, query prefix, provenance
```

`<commit>` is the analysed commit (the PR head, not GitHub's synthetic merge commit).

## How calls are resolved

| `resolution` | example | how |
|---|---|---|
| `module` | `helper(x)` | name defined in the same module |
| `import` | `httpx.Client(...)`, `from ._models import Request` | import table, following re-exports and relative imports |
| `local` | `inner()` | nested function in an enclosing scope |
| `self` | `self._send_single_request(...)` | method lookup on the enclosing class and its bases |
| `super` | `super().__init__(...)` | lookup starting at the base classes |
| `type` | `response.close()` with `response: Response` | parameter/return annotations, `x = Cls(...)`, `self.attr = ...`, properties |
| `constructor` | `Headers(...)` | extra edge to the class's `__init__` (edge kind `instantiate` goes to the class) |
| `external` / `builtin` | `os.path.join`, `len` | targets outside the analysed tree |

Calls on values whose type can't be determined statically (results of
external factories, loop variables, `getattr`, callbacks) are counted as
unresolved and omitted unless `--include-unresolved` is passed. Method
calls bind to the statically known class, so `self.prepare()` in a base
class points at the base implementation, not subclass overrides.

## graph.json

networkx `node_link` data (the swegemma format): a directed multigraph that
loads directly with `networkx.node_link_graph(data, edges="edges")`. (Examples below are from httpx.)

```jsonc
{
  "directed": true,
  "multigraph": true,
  "graph": {
    "generator": "...", "schema": 3, "root": "httpx", "commit": "<sha>", "options": { ... },
    "stats": { "classes": 87, "methods": 361, "call_sites": 1559, "resolved_ratio": 0.84, ... }
  },
  "nodes": [
    { "id": "httpx._client.Client.send", "name": "send", "text": "    def send(\n        self,\n ...",
      "kind": "method", "qualname": "Client.send", "module": "httpx._client", "file": "httpx/_client.py",
      "lineno": 879, "end_lineno": 928, "text_lineno": 879, "parent": "httpx._client.Client", "async": false },
    { "id": "os.path.join", "name": "join", "text": "", "kind": "external" }
  ],
  "edges": [
    { "source": "httpx._client.Client.send", "target": "httpx._models.Response.close",
      "key": 0, "type": "calls", "resolution": "type", "count": 1, "lines": [927] }
  ]
}
```

Every node has `id`, `name`, `text`; the other node fields are extra
attributes. Edge `type` is `calls`, or `instantiates` for `Cls(...)` (which
also produces a `calls` edge to `Cls.__init__`). Edges are aggregated per
(source, target, type), so `key` is the networkx multigraph key (0, 1, ...
per source/target pair); `lines` lists every call site. Every edge endpoint is
a node. `--edges-key links` writes the edge list under `links` instead (the
default key of networkx < 3.4).

Node kinds: `module`, `class`, `function`, `method`, plus `external`,
`builtin`, `unresolved` for call targets outside the tree.

`text` is the node's source, sliced verbatim from the file (original
indentation, decorators included): lines `text_lineno`..`end_lineno`. A
module's text is the whole file; a class's text includes its methods. A
name defined more than once in one scope (property getter + setter,
`@overload` stubs, one `def` per `if`/`elif` branch) is one node: its text is
each definition's own source joined by a blank line, and `text_spans` lists
the `[first, last]` lines of each, so code between the definitions is never
included. Empty modules (blank `__init__.py`) have `"text": ""`. Nodes outside the tree have `"text": ""`.
`--no-text` omits the field.

## Embeddings

`embed.py` encodes each node's `text` with
[`Snowflake/snowflake-arctic-embed-m-v1.5`](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5)
(`--model` to change), keeps the first 256 components (the model is trained
for this Matryoshka truncation) and L2-normalises, so cosine = dot product.
To search, embed the query with the same model, prefix
`Represent this sentence for searching relevant passages: `, truncate to 256
and normalise. Inputs longer than the model's 512-token window are truncated,
so large modules/classes are represented by their beginning.

`check_outputs.py` verifies keys == nodes with text, float32[256], unit norm,
and a semantic smoke test: same-named methods in different classes
(`Client.get` / `AsyncClient.get`) must be clearly closer than random pairs.

Flags: `--no-external`, `--include-builtins`, `--include-unresolved`,
`--exclude DIR`, `--root` (import root for module names).
