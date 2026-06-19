# Ascend/CANN SQLite RAG

This repository contains a local hybrid RAG indexer for multiple corpora.

## Dataset Model

The builder is now split into:

- a shared indexing engine
- per-source adapters (`asc-devkit`, `cann-docs`, and future source kinds)
- a dataset manifest that defines which sources are merged into one SQLite database

Default manifest: `hw_rag.dataset.json`

Current default dataset:

- `docs/asc-devkit`
- `docs/cann-docs`

## Build

Build the default dataset with deterministic offline embeddings:

```powershell
python hw_rag_cli.py build --embedding-provider hash
```

Build a specific dataset manifest:

```powershell
python hw_rag_cli.py build --dataset ./hw_rag.dataset.json --db ./hw_rag.sqlite --embedding-provider hash
```

Build a single source explicitly:

```powershell
python hw_rag_cli.py build --source ./docs/asc-devkit --source-kind asc-devkit --db ./hw_rag.sqlite --embedding-provider hash
python hw_rag_cli.py build --source ./docs/cann-docs --source-kind cann-docs --path-prefix cann-docs --db ./hw_rag.sqlite --embedding-provider hash
```

OpenAI embeddings:

```powershell
$env:OPENAI_API_KEY = "..."
python hw_rag_cli.py build --embedding-provider openai
```

## Search

```powershell
python hw_rag_cli.py search "DataCopy GM UB LocalTensor" --db ./hw_rag.sqlite --limit 8
python hw_rag_cli.py search "TQue AllocTensor FreeTensor TPipe" --db ./hw_rag.sqlite --limit 8
python hw_rag_cli.py search "aclnnAddRmsNorm" --db ./hw_rag.sqlite --limit 8
```

Optional corpus filter:

```powershell
python hw_rag_cli.py search "DataCopy" --type api_reference
python hw_rag_cli.py search "AllocTensor" --type headers
```

Search results now include `corpus=<source_name>/<source_kind>` so you can tell whether a hit came from `asc-devkit` or `cann-docs`.

## Manifest Shape

```json
{
  "name": "ascend-cann-hybrid",
  "db_path": "hw_rag.sqlite",
  "sources": [
    {
      "name": "asc-devkit",
      "kind": "asc-devkit",
      "root": "docs/asc-devkit",
      "include_tools": true,
      "path_prefix": "asc-devkit"
    },
    {
      "name": "cann-docs",
      "kind": "cann-docs",
      "root": "docs/cann-docs",
      "include_tools": true,
      "path_prefix": "cann-docs"
    }
  ]
}
```

Per-source behavior is controlled by adapter logic, not by separate top-level scripts. That keeps build/search generic while still allowing different classification and skip rules per source kind.

## Database Contents

- `documents`: one row per indexed file, including `source_name`, `source_kind`, and `relative_path`
- `chunks`: retrievable text/code chunks with line ranges and source metadata
- `chunk_fts`: SQLite FTS5 index for BM25 keyword retrieval
- `embeddings`: vector blobs for semantic retrieval
- `symbols`: extracted symbols from code-like files, tagged by source
- `corpus_meta`: schema version, embedder, dataset manifest metadata, and build stats

## MCP Integration

The MCP server exposes:

- `ascend_rag_search`
- `ascend_rag_read`
- `ascend_rag_context`

Smoke test:

```powershell
python scripts/test_mcp_server.py
```

Codex:

```powershell
codex mcp add ascend-c-rag --env PYTHONIOENCODING=utf-8 --env PYTHONPATH=/absolute/path/to/repo -- python -m hw_rag.mcp_server --db /absolute/path/to/repo/hw_rag.sqlite --embedding-provider hash
```

Claude Code:

```powershell
claude mcp add ascend-c-rag -e PYTHONIOENCODING=utf-8 -e PYTHONPATH=/absolute/path/to/repo -- python -m hw_rag.mcp_server --db /absolute/path/to/repo/hw_rag.sqlite --embedding-provider hash
```

Use an absolute `--db` path for MCP registration. Some MCP clients launch stdio subprocesses without a stable working directory, and a relative SQLite path can silently resolve to the wrong file.
