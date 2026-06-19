from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from .corpus import (
    LEGACY_DEFAULT_DB,
    DatasetConfig,
    SourceConfig,
    dataset_to_metadata,
    default_db_path,
    load_dataset_manifest,
    single_source_dataset,
)
from .db import connect, init_db
from .embeddings import Embedder, make_embedder, vector_to_blob
from .plugins import get_plugin


DEFAULT_DB = LEGACY_DEFAULT_DB


def build_database(
    source: Path,
    db_path: Path,
    embedder: Embedder,
    include_tools: bool = True,
) -> None:
    dataset = single_source_dataset(
        source,
        include_tools=include_tools,
        path_prefix="",
        dataset_name=source.resolve().name,
    )
    build_dataset(dataset, db_path, embedder)


def build_dataset(
    dataset: DatasetConfig,
    db_path: Path | None,
    embedder: Embedder,
) -> Path:
    dataset = dataset.resolved()
    resolved_db_path = resolve_db_path(dataset, db_path)
    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(resolved_db_path) as conn:
        init_db(conn)
        conn.execute(
            "INSERT INTO corpus_meta(key, value) VALUES (?, ?)",
            ("dataset_name", dataset.name),
        )
        conn.execute(
            "INSERT INTO corpus_meta(key, value) VALUES (?, ?)",
            ("dataset", json.dumps(dataset_to_metadata(dataset), ensure_ascii=False, sort_keys=True)),
        )
        if len(dataset.sources) == 1:
            conn.execute(
                "INSERT INTO corpus_meta(key, value) VALUES (?, ?)",
                ("source", str(dataset.sources[0].root)),
            )
        conn.execute(
            "INSERT INTO corpus_meta(key, value) VALUES (?, ?)",
            ("embedder", embedder.name),
        )
        stats = ingest_dataset(conn, dataset, embedder)
        conn.execute(
            "INSERT INTO corpus_meta(key, value) VALUES (?, ?)",
            ("stats", json.dumps(stats, ensure_ascii=False, sort_keys=True)),
        )
        conn.commit()
    return resolved_db_path


def build_manifest(
    manifest_path: Path,
    db_path: Path | None,
    embedder: Embedder,
) -> Path:
    dataset = load_dataset_manifest(manifest_path)
    return build_dataset(dataset, db_path, embedder)


def ingest_dataset(
    conn: sqlite3.Connection,
    dataset: DatasetConfig,
    embedder: Embedder,
) -> dict[str, object]:
    total = {"documents": 0, "chunks": 0, "symbols": 0, "skipped": 0}
    by_source: dict[str, dict[str, int]] = {}
    for source in dataset.sources:
        source_stats = ingest_source(conn, source, embedder)
        by_source[source.name] = source_stats
        for key in total:
            total[key] += source_stats[key]
    return {"total": total, "sources": by_source}


def ingest_source(
    conn: sqlite3.Connection,
    source: SourceConfig,
    embedder: Embedder,
) -> dict[str, int]:
    stats = {"documents": 0, "chunks": 0, "symbols": 0, "skipped": 0}
    plugin = get_plugin(source.kind)
    for path in sorted(source.root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(source.root).as_posix()
        if plugin.should_skip(relative_path, path, source.include_tools):
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

        corpus_type, topic, role = plugin.classify_path(relative_path)
        sha = hashlib.sha256(raw).hexdigest()
        storage_path = source.storage_path_for(relative_path)
        cursor = conn.execute(
            """
            INSERT INTO documents(
                path, relative_path, abs_path, sha256, size_bytes, extension,
                source_name, source_kind, source_root, corpus_type, topic, role
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                storage_path,
                relative_path,
                str(path),
                sha,
                len(raw),
                path.suffix.lower(),
                source.name,
                source.kind,
                str(source.root),
                corpus_type,
                topic,
                role,
            ),
        )
        document_id = int(cursor.lastrowid)
        stats["documents"] += 1

        chunks = plugin.chunk_text(relative_path, text)
        for chunk_index, chunk in enumerate(chunks):
            chunk_cursor = conn.execute(
                """
                INSERT INTO chunks(
                    document_id, chunk_index, title, content, start_line, end_line,
                    source_name, source_kind, corpus_type, topic, path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    chunk_index,
                    chunk.title,
                    chunk.content,
                    chunk.start_line,
                    chunk.end_line,
                    source.name,
                    source.kind,
                    corpus_type,
                    topic,
                    storage_path,
                ),
            )
            chunk_id = int(chunk_cursor.lastrowid)
            vector = embedder.embed(f"{storage_path}\n{chunk.title}\n{chunk.content[:12000]}")
            conn.execute(
                """
                INSERT INTO embeddings(chunk_id, provider, dimension, vector)
                VALUES (?, ?, ?, ?)
                """,
                (chunk_id, embedder.name, embedder.dimension, vector_to_blob(vector)),
            )
            stats["chunks"] += 1

        if plugin.should_extract_symbols(path):
            for symbol in plugin.extract_symbols(text):
                conn.execute(
                    """
                    INSERT INTO symbols(
                        document_id, path, source_name, source_kind, name, kind,
                        signature, line, corpus_type, topic
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        storage_path,
                        source.name,
                        source.kind,
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


def resolve_db_path(dataset: DatasetConfig, db_path: Path | None) -> Path:
    if db_path is not None:
        return db_path.resolve()
    if dataset.db_path is not None:
        return dataset.db_path.resolve()
    return default_db_path(Path.cwd())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local SQLite RAG index.")
    parser.add_argument("--dataset", type=Path, help="Path to a dataset manifest JSON file.")
    parser.add_argument("--source", type=Path, help="Build from a single source root.")
    parser.add_argument("--source-name", help="Logical source name for single-source builds.")
    parser.add_argument("--source-kind", help="Source kind override, e.g. asc-devkit or cann-docs.")
    parser.add_argument("--path-prefix", help="Storage path prefix for single-source builds.")
    parser.add_argument("--db", type=Path)
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
    if args.dataset:
        db_path = build_manifest(args.dataset, args.db, embedder)
        print(f"Built {db_path} from dataset {args.dataset} using {embedder.name}")
        return
    if not args.source:
        raise SystemExit("Either --dataset or --source is required.")
    dataset = single_source_dataset(
        args.source,
        name=args.source_name,
        kind=args.source_kind,
        include_tools=not args.exclude_tools,
        path_prefix=args.path_prefix or "",
        dataset_name=args.source_name or args.source.resolve().name,
    )
    db_path = build_dataset(dataset, args.db, embedder)
    print(f"Built {db_path} from {args.source} using {embedder.name}")


if __name__ == "__main__":
    main()
