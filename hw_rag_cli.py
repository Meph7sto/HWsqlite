from __future__ import annotations

import argparse
import sys

from hw_rag import build as build_module
from hw_rag import search as search_module


def main() -> None:
    configure_stdout()
    parser = argparse.ArgumentParser(description="Ascend C local SQLite RAG tool.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build the SQLite index.")
    build_parser.add_argument("--source", type=build_module.Path, default=build_module.Path("asc-devkit"))
    build_parser.add_argument("--db", type=build_module.Path, default=build_module.DEFAULT_DB)
    build_parser.add_argument("--embedding-provider", choices=["auto", "hash", "openai"], default="auto")
    build_parser.add_argument("--exclude-tools", action="store_true")

    search_parser = subparsers.add_parser("search", help="Search the SQLite index.")
    search_parser.add_argument("query")
    search_parser.add_argument("--db", type=search_module.Path, default=search_module.Path("hw_rag.sqlite"))
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument("--type", dest="type_filter")
    search_parser.add_argument("--embedding-provider", choices=["auto", "hash", "openai"], default="auto")

    args = parser.parse_args()
    if args.command == "build":
        embedder = build_module.make_embedder(args.embedding_provider)
        build_module.build_database(
            args.source,
            args.db,
            embedder,
            include_tools=not args.exclude_tools,
        )
        print(f"Built {args.db} from {args.source} using {embedder.name}")
    elif args.command == "search":
        results = search_module.hybrid_search(
            db_path=args.db,
            query=args.query,
            limit=args.limit,
            type_filter=args.type_filter,
            embedding_provider=args.embedding_provider,
        )
        for index, result in enumerate(results, start=1):
            location = f"{result.path}:{result.start_line}"
            safe_print(f"{index}. [{result.score:.3f}] {result.corpus_type}/{result.topic} {location}")
            if result.title:
                safe_print(f"   {result.title}")
            safe_print(f"   source={result.source}")
            safe_print(f"   {result.snippet.replace(chr(10), ' ')[:300]}")


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()
