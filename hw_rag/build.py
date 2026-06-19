from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from .chunking import chunk_text
from .classifier import CODE_EXTENSIONS, classify_path, is_text_file
from .db import connect, init_db
from .embeddings import Embedder, make_embedder, vector_to_blob
from .symbols import extract_symbols


DEFAULT_DB = Path("hw_rag.sqlite")


def build_database(
    source: Path,
    db_path: Path,
    embedder: Embedder,
    include_tools: bool = True,
) -> None:
    source = source.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        init_db(conn)
        conn.execute("INSERT INTO corpus_meta(key, value) VALUES (?, ?)", ("source", str(source)))
        conn.execute(
            "INSERT INTO corpus_meta(key, value) VALUES (?, ?)",
            ("embedder", embedder.name),
        )
        stats = ingest_source(conn, source, embedder, include_tools)
        conn.execute(
            "INSERT INTO corpus_meta(key, value) VALUES (?, ?)",
            ("stats", json.dumps(stats, ensure_ascii=False, sort_keys=True)),
        )
        conn.commit()


def ingest_source(
    conn: sqlite3.Connection,
    source: Path,
    embedder: Embedder,
    include_tools: bool,
) -> dict[str, int]:
    stats = {"documents": 0, "chunks": 0, "symbols": 0, "skipped": 0}
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(source).as_posix()
        if should_skip(relative_path, path, include_tools):
            stats["skipped"] += 1
            continue
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("gb18030")
            except UnicodeDecodeError:
                stats["skipped"] += 1
                continue

        corpus_type, topic, role = classify_path(relative_path)
        sha = hashlib.sha256(raw).hexdigest()
        cursor = conn.execute(
            """
            INSERT INTO documents(path, abs_path, sha256, size_bytes, extension, corpus_type, topic, role)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                relative_path,
                str(path),
                sha,
                len(raw),
                path.suffix.lower(),
                corpus_type,
                topic,
                role,
            ),
        )
        document_id = int(cursor.lastrowid)
        stats["documents"] += 1

        chunks = chunk_text(relative_path, text)
        for chunk_index, chunk in enumerate(chunks):
            chunk_cursor = conn.execute(
                """
                INSERT INTO chunks(
                    document_id, chunk_index, title, content, start_line, end_line,
                    corpus_type, topic, path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    chunk_index,
                    chunk.title,
                    chunk.content,
                    chunk.start_line,
                    chunk.end_line,
                    corpus_type,
                    topic,
                    relative_path,
                ),
            )
            chunk_id = int(chunk_cursor.lastrowid)
            vector = embedder.embed(f"{relative_path}\n{chunk.title}\n{chunk.content[:12000]}")
            conn.execute(
                """
                INSERT INTO embeddings(chunk_id, provider, dimension, vector)
                VALUES (?, ?, ?, ?)
                """,
                (chunk_id, embedder.name, embedder.dimension, vector_to_blob(vector)),
            )
            stats["chunks"] += 1

        if path.suffix.lower() in CODE_EXTENSIONS:
            for symbol in extract_symbols(text):
                conn.execute(
                    """
                    INSERT INTO symbols(
                        document_id, path, name, kind, signature, line, corpus_type, topic
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        relative_path,
                        symbol.name,
                        symbol.kind,
                        symbol.signature,
                        symbol.line,
                        corpus_type,
                        topic,
                    ),
                )
                stats["symbols"] += 1
    return stats


def should_skip(relative_path: str, path: Path, include_tools: bool) -> bool:
    normalized = relative_path.replace("\\", "/")
    if "/figures/" in f"/{normalized}" or path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return True
    if any(part in {".git", "__pycache__", "build", "out"} for part in Path(relative_path).parts):
        return True
    if not include_tools and normalized.startswith(("tools/", "scripts/", "tests/")):
        return True
    return not is_text_file(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Ascend C SQLite RAG index.")
    parser.add_argument("--source", type=Path, default=Path("asc-devkit"))
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--embedding-provider",
        choices=["auto", "hash", "openai"],
        default="auto",
        help="auto uses OpenAI when OPENAI_API_KEY is set, otherwise hashing fallback.",
    )
    parser.add_argument("--exclude-tools", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedder = make_embedder(args.embedding_provider)
    build_database(args.source, args.db, embedder, include_tools=not args.exclude_tools)
    print(f"Built {args.db} from {args.source} using {embedder.name}")


if __name__ == "__main__":
    main()

