# HWsqlite

SQLite-backed hybrid RAG index for Huawei Ascend C / `asc-devkit`.

This repo contains:

- `hw_rag.sqlite`: prebuilt SQLite index, stored with Git LFS.
- `hw_rag/`: ingestion, classification, chunking, FTS5, embedding, search, and MCP server code.
- `hw_rag_cli.py`: command-line build/search entrypoint.
- `scripts/test_mcp_server.py`: MCP smoke test.
- `tests/`: basic regression tests.
- `HW_RAG_README.md`: usage details.

## Quick Search

```powershell
python hw_rag_cli.py search "DataCopy GM UB LocalTensor" --db hw_rag.sqlite --limit 8
python hw_rag_cli.py search "TQue AllocTensor FreeTensor TPipe" --db hw_rag.sqlite --limit 8
```

## MCP

```powershell
python scripts/test_mcp_server.py
```

Codex:

```powershell
codex mcp add ascend-c-rag --env PYTHONIOENCODING=utf-8 -- python -m hw_rag.mcp_server --db W:\DB\HW\hw_rag.sqlite --embedding-provider hash
```

Claude Code:

```powershell
claude mcp add ascend-c-rag -e PYTHONIOENCODING=utf-8 -- python -m hw_rag.mcp_server --db W:\DB\HW\hw_rag.sqlite --embedding-provider hash
```

See `HW_RAG_README.md` for rebuild and integration details.

