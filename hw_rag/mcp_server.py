from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .db import connect
from .search import hybrid_search


JSONRPC_VERSION = "2.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCP server for the Ascend C SQLite RAG index.")
    parser.add_argument("--db", type=Path, default=Path("hw_rag.sqlite"))
    parser.add_argument("--embedding-provider", choices=["auto", "hash", "openai"], default="auto")
    return parser.parse_args()


class AscendRagMcpServer:
    def __init__(self, db_path: Path, embedding_provider: str) -> None:
        self.db_path = db_path
        self.embedding_provider = embedding_provider

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        request_id = message.get("id")
        try:
            if method == "initialize":
                return self.response(request_id, self.initialize_result())
            if method == "notifications/initialized":
                return None
            if method == "tools/list":
                return self.response(request_id, {"tools": self.tools()})
            if method == "tools/call":
                params = message.get("params") or {}
                return self.response(request_id, self.call_tool(params))
            return self.error(request_id, -32601, f"Unknown method: {method}")
        except Exception as exc:
            return self.error(request_id, -32603, str(exc))

    def initialize_result(self) -> dict[str, Any]:
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": "ascend-c-rag",
                "version": "0.1.0",
            },
            "instructions": (
                "Use this server before writing Ascend C, CANN, SIMD/SIMT, "
                "TPipe/TQue, DataCopy, tiling, or Ascend CMake code. Prefer "
                "api_reference, headers, and examples over implementation internals."
            ),
        }

    def tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "ascend_rag_search",
                "description": (
                    "Hybrid search over asc-devkit: API docs, headers, examples, guides, "
                    "tests, implementation, and build files."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 8, "minimum": 1, "maximum": 30},
                        "type_filter": {
                            "type": "string",
                            "description": "Optional corpus type, e.g. api_reference, headers, examples, guide.",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "ascend_rag_read",
                "description": "Read a chunk or file range from the indexed asc-devkit corpus.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "default": 1, "minimum": 1},
                        "line_count": {"type": "integer", "default": 120, "minimum": 1, "maximum": 500},
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "ascend_rag_context",
                "description": (
                    "Build a compact context pack for code generation. It returns prioritized "
                    "API docs, headers, examples, guide notes, and warnings."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "limit": {"type": "integer", "default": 12, "minimum": 4, "maximum": 30},
                    },
                    "required": ["task"],
                },
            },
        ]

    def call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name == "ascend_rag_search":
            return self.tool_search(arguments)
        if name == "ascend_rag_read":
            return self.tool_read(arguments)
        if name == "ascend_rag_context":
            return self.tool_context(arguments)
        raise ValueError(f"Unknown tool: {name}")

    def tool_search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments["query"])
        limit = int(arguments.get("limit", 8))
        type_filter = arguments.get("type_filter")
        results = hybrid_search(
            db_path=self.db_path,
            query=query,
            limit=limit,
            type_filter=type_filter,
            embedding_provider=self.embedding_provider,
        )
        payload = [
            {
                "path": result.path,
                "title": result.title,
                "corpus_type": result.corpus_type,
                "topic": result.topic,
                "start_line": result.start_line,
                "end_line": result.end_line,
                "score": round(result.score, 4),
                "source": result.source,
                "snippet": result.snippet,
            }
            for result in results
        ]
        return text_content(json.dumps(payload, ensure_ascii=False, indent=2))

    def tool_read(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = str(arguments["path"]).replace("\\", "/")
        start_line = int(arguments.get("start_line", 1))
        line_count = int(arguments.get("line_count", 120))
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT abs_path FROM documents WHERE path = ?",
                (path,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Path is not indexed: {path}")
        abs_path = Path(row["abs_path"])
        text = read_file_range(abs_path, start_line, line_count)
        return text_content(f"{path}:{start_line}\n{text}")

    def tool_context(self, arguments: dict[str, Any]) -> dict[str, Any]:
        task = str(arguments["task"])
        limit = int(arguments.get("limit", 12))
        results = hybrid_search(
            db_path=self.db_path,
            query=task,
            limit=limit,
            embedding_provider=self.embedding_provider,
        )
        grouped: dict[str, list[Any]] = {}
        for result in results:
            grouped.setdefault(result.corpus_type, []).append(result)

        lines = [
            f"Task: {task}",
            "",
            "Use order: api_reference -> headers -> examples -> guide -> tests -> implementation.",
            "Do not invent Ascend C APIs; verify names and signatures against headers/API docs.",
            "",
        ]
        for corpus_type in [
            "api_reference",
            "headers",
            "examples",
            "guide",
            "tests",
            "implementation",
            "build_system",
        ]:
            items = grouped.get(corpus_type, [])
            if not items:
                continue
            lines.append(f"## {corpus_type}")
            for item in items[:5]:
                title = f" - {item.title}" if item.title else ""
                lines.append(
                    f"- {item.path}:{item.start_line}-{item.end_line}{title} "
                    f"[{item.source}, score={item.score:.3f}]"
                )
                snippet = " ".join(item.snippet.split())[:500]
                if snippet:
                    lines.append(f"  {snippet}")
            lines.append("")
        return text_content("\n".join(lines).strip())

    @staticmethod
    def response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}

    @staticmethod
    def error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "error": {"code": code, "message": message},
        }


def text_content(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def read_file_range(path: Path, start_line: int, line_count: int) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="gb18030")
    lines = text.splitlines()
    start_index = max(0, start_line - 1)
    end_index = min(len(lines), start_index + line_count)
    return "\n".join(
        f"{line_number}: {line}"
        for line_number, line in enumerate(lines[start_index:end_index], start=start_index + 1)
    )


def run_stdio(server: AscendRagMcpServer) -> None:
    configure_stdio()
    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        message = json.loads(raw_line)
        response = server.handle(message)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


def configure_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", newline="\n")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace", newline="\n")


def main() -> None:
    args = parse_args()
    server = AscendRagMcpServer(args.db, args.embedding_provider)
    run_stdio(server)


if __name__ == "__main__":
    main()

