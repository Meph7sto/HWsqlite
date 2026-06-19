from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_VERSION = 1


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS chunk_fts;
        DROP TABLE IF EXISTS embeddings;
        DROP TABLE IF EXISTS symbols;
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS documents;
        DROP TABLE IF EXISTS corpus_meta;

        CREATE TABLE corpus_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            abs_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            extension TEXT NOT NULL,
            corpus_type TEXT NOT NULL,
            topic TEXT NOT NULL,
            role TEXT NOT NULL
        );

        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            corpus_type TEXT NOT NULL,
            topic TEXT NOT NULL,
            path TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE chunk_fts USING fts5(
            title,
            content,
            path UNINDEXED,
            corpus_type UNINDEXED,
            topic UNINDEXED,
            content='chunks',
            content_rowid='id',
            tokenize='unicode61 tokenchars ''_./-'''
        );

        CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunk_fts(rowid, title, content, path, corpus_type, topic)
            VALUES (new.id, new.title, new.content, new.path, new.corpus_type, new.topic);
        END;

        CREATE TABLE embeddings (
            chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
            provider TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            vector BLOB NOT NULL
        );

        CREATE TABLE symbols (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            path TEXT NOT NULL,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            signature TEXT NOT NULL,
            line INTEGER NOT NULL,
            corpus_type TEXT NOT NULL,
            topic TEXT NOT NULL
        );

        CREATE INDEX idx_documents_type_topic ON documents(corpus_type, topic);
        CREATE INDEX idx_chunks_type_topic ON chunks(corpus_type, topic);
        CREATE INDEX idx_symbols_name ON symbols(name);
        CREATE INDEX idx_symbols_kind ON symbols(kind);
        """
    )
    conn.execute(
        "INSERT INTO corpus_meta(key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )

