from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def send(process: subprocess.Popen[str], message: dict) -> dict:
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("MCP server returned no response")
    return json.loads(line)


def main() -> None:
    command = [
        sys.executable,
        "-m",
        "hw_rag.mcp_server",
        "--db",
        str(ROOT / "hw_rag.sqlite"),
        "--embedding-provider",
        "hash",
    ]
    with subprocess.Popen(
        command,
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    ) as process:
        init = send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "smoke-test", "version": "0"},
                },
            },
        )
        tools = send(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        result = send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "ascend_rag_search",
                    "arguments": {"query": "DataCopy LocalTensor", "limit": 2},
                },
            },
        )
        process.terminate()

    assert init["result"]["serverInfo"]["name"] == "ascend-c-rag"
    tool_names = {tool["name"] for tool in tools["result"]["tools"]}
    assert "ascend_rag_search" in tool_names
    assert "DataCopy" in result["result"]["content"][0]["text"]
    print("mcp smoke test ok")


if __name__ == "__main__":
    main()
