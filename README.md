# HWsqlite

SQLite-backed hybrid RAG index for Huawei Ascend C and CANN documentation.

This repo contains:

- `hw_rag.sqlite`: prebuilt SQLite index, stored with Git LFS.
- `hw_rag.dataset.json`: dataset manifest defining the current multi-source corpus.
- `hw_rag/`: ingestion, classification, chunking, FTS5, embedding, search, and MCP server code.
- `hw_rag_cli.py`: command-line build/search entrypoint.
- `scripts/test_mcp_server.py`: MCP smoke test.
- `tests/`: basic regression tests.
- `HW_RAG_README.md`: usage details.

The default dataset currently combines:

- `docs/asc-devkit`
- `docs/cann-docs`

## Quick Search

```powershell
python hw_rag_cli.py search "DataCopy GM UB LocalTensor" --db hw_rag.sqlite --limit 8
python hw_rag_cli.py search "TQue AllocTensor FreeTensor TPipe" --db hw_rag.sqlite --limit 8
python hw_rag_cli.py search "aclnnAddRmsNorm" --db hw_rag.sqlite --limit 8
```

## MCP

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

Use an absolute `--db` path for MCP servers. Some clients do not guarantee the subprocess working directory, and a relative SQLite path can silently point at a different file.

See `HW_RAG_README.md` for rebuild and integration details.
