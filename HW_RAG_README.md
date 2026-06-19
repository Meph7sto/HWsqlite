# Ascend C SQLite RAG

This workspace contains a local hybrid RAG indexer for `asc-devkit`.

## Build

Offline deterministic embedding:

```powershell
python hw_rag_cli.py build --source asc-devkit --db hw_rag.sqlite --embedding-provider hash
```

OpenAI embedding, stored in the same SQLite database:

```powershell
$env:OPENAI_API_KEY = "..."
python hw_rag_cli.py build --source asc-devkit --db hw_rag.sqlite --embedding-provider openai
```

## Search

```powershell
python hw_rag_cli.py search "DataCopy GM UB LocalTensor" --db hw_rag.sqlite --limit 8
python hw_rag_cli.py search "TQue AllocTensor FreeTensor TPipe" --db hw_rag.sqlite --limit 8
python hw_rag_cli.py search "通过 CMake 编译 Ascend C 算子 npu-arch" --db hw_rag.sqlite --limit 8
```

Optional corpus filter:

```powershell
python hw_rag_cli.py search "DataCopy" --type api_reference
python hw_rag_cli.py search "DataCopy" --type examples
python hw_rag_cli.py search "AllocTensor" --type headers
```

## Database Contents

- `documents`: one row per indexed file.
- `chunks`: retrievable text/code chunks with path and line numbers.
- `chunk_fts`: SQLite FTS5 index for BM25 keyword retrieval.
- `embeddings`: vector BLOBs for semantic retrieval.
- `symbols`: extracted C/C++/Python/CMake symbols from code-like files.
- `corpus_meta`: source path, embedder, schema version, and build stats.

Hybrid search combines exact API title/path matches, FTS5, symbol matches, and vector cosine reranking. Public API docs, headers, and examples are weighted above implementation internals.

## MCP Integration

The MCP server exposes three tools:

- `ascend_rag_search`: hybrid search over the SQLite index.
- `ascend_rag_read`: read indexed files by path and line range.
- `ascend_rag_context`: build a compact code-generation context pack.

Smoke test:

```powershell
python scripts/test_mcp_server.py
```

Codex:

```powershell
codex mcp add ascend-c-rag --env PYTHONIOENCODING=utf-8 -- C:\Users\ASUS\miniconda3\python.exe -m hw_rag.mcp_server --db W:\DB\HW\hw_rag.sqlite --embedding-provider hash
codex mcp list
```

Claude Code:

```powershell
claude mcp add ascend-c-rag -e PYTHONIOENCODING=utf-8 -- C:\Users\ASUS\miniconda3\python.exe -m hw_rag.mcp_server --db W:\DB\HW\hw_rag.sqlite --embedding-provider hash
claude mcp list
```

After adding the server, start a new Codex or Claude Code session. Ask the agent to use `ascend_rag_context` before writing Ascend C code.
