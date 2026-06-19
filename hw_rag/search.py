from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from .db import connect
from .embeddings import blob_to_vector, cosine_similarity, make_embedder


@dataclass
class SearchResult:
    chunk_id: int
    path: str
    title: str
    corpus_type: str
    topic: str
    start_line: int
    end_line: int
    score: float
    source: str
    snippet: str


def hybrid_search(
    db_path: Path,
    query: str,
    limit: int = 10,
    type_filter: str | None = None,
    embedding_provider: str = "auto",
) -> list[SearchResult]:
    with connect(db_path) as conn:
        fts = _fts_search(conn, query, limit * 4, type_filter)
        symbols = _symbol_search(conn, query, limit * 3, type_filter)
        api_exact = _api_exact_search(conn, query, limit * 2, type_filter)
        vector = _vector_search(conn, query, limit * 4, type_filter, embedding_provider)

    merged: dict[int, SearchResult] = {}
    for weight, source_results in (
        (1.8, api_exact),
        (1.0, fts),
        (1.25, symbols),
        (0.9, vector),
    ):
        for rank, result in enumerate(source_results):
            rank_score = (weight * corpus_priority(result.corpus_type)) / (rank + 1)
            existing = merged.get(result.chunk_id)
            if existing is None:
                result.score = rank_score
                merged[result.chunk_id] = result
            else:
                existing.score += rank_score
                if source_results is symbols:
                    existing.source = f"{existing.source}+symbol"
    return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:limit]


def corpus_priority(corpus_type: str) -> float:
    priorities = {
        "api_reference": 1.45,
        "api_catalog": 1.35,
        "headers": 1.35,
        "examples": 1.25,
        "guide": 1.0,
        "tests": 0.85,
        "build_system": 0.8,
        "tools": 0.75,
        "implementation": 0.65,
    }
    return priorities.get(corpus_type, 0.7)


def _fts_search(conn, query: str, limit: int, type_filter: str | None) -> list[SearchResult]:
    where = "chunk_fts MATCH ?"
    params: list[object] = [to_fts_query(query)]
    if type_filter:
        where += " AND c.corpus_type = ?"
        params.append(type_filter)
    rows = conn.execute(
        f"""
        SELECT c.id, c.path, c.title, c.corpus_type, c.topic, c.start_line, c.end_line,
               snippet(chunk_fts, 1, '[', ']', '...', 18) AS snippet,
               bm25(chunk_fts) AS bm25_score
        FROM chunk_fts
        JOIN chunks c ON c.id = chunk_fts.rowid
        WHERE {where}
        ORDER BY bm25_score
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    return [
        SearchResult(
            chunk_id=row["id"],
            path=row["path"],
            title=row["title"],
            corpus_type=row["corpus_type"],
            topic=row["topic"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            score=0.0,
            source="fts",
            snippet=row["snippet"] or "",
        )
        for row in rows
    ]


def _api_exact_search(conn, query: str, limit: int, type_filter: str | None) -> list[SearchResult]:
    terms = extract_identifier_terms(query)
    if not terms:
        return []
    clauses = []
    params: list[object] = []
    for term in terms[:8]:
        clauses.append("(c.title = ? OR c.path LIKE ?)")
        params.extend([term, f"%/{term}.md"])
    where = "(" + " OR ".join(clauses) + ")"
    if type_filter:
        where += " AND c.corpus_type = ?"
        params.append(type_filter)
    rows = conn.execute(
        f"""
        SELECT c.id, c.path, c.title, c.corpus_type, c.topic, c.start_line, c.end_line,
               substr(c.content, 1, 500) AS snippet
        FROM chunks c
        WHERE {where}
        ORDER BY
            CASE c.corpus_type
                WHEN 'api_reference' THEN 0
                WHEN 'api_catalog' THEN 1
                WHEN 'headers' THEN 2
                WHEN 'examples' THEN 3
                ELSE 4
            END,
            length(c.path)
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    return [
        SearchResult(
            chunk_id=row["id"],
            path=row["path"],
            title=row["title"],
            corpus_type=row["corpus_type"],
            topic=row["topic"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            score=0.0,
            source="api-exact",
            snippet=row["snippet"] or "",
        )
        for row in rows
    ]


def _symbol_search(conn, query: str, limit: int, type_filter: str | None) -> list[SearchResult]:
    terms = strong_symbol_terms(query)
    if not terms:
        return []
    clauses = []
    params: list[object] = []
    for term in terms[:6]:
        clauses.append("(s.name = ? OR s.name LIKE ?)")
        params.extend([term, f"{term}%"])
    where = "(" + " OR ".join(clauses) + ")"
    if type_filter:
        where += " AND c.corpus_type = ?"
        params.append(type_filter)
    rows = conn.execute(
        f"""
        SELECT c.id, c.path, c.title, c.corpus_type, c.topic, c.start_line, c.end_line,
               s.name, s.kind, s.signature, s.line
        FROM symbols s
        JOIN chunks c ON c.document_id = s.document_id
            AND s.line BETWEEN c.start_line AND c.end_line
        WHERE {where}
        GROUP BY c.id
        ORDER BY
            CASE c.corpus_type
                WHEN 'headers' THEN 0
                WHEN 'api_reference' THEN 1
                WHEN 'examples' THEN 2
                WHEN 'tests' THEN 3
                WHEN 'implementation' THEN 4
                ELSE 5
            END,
            CASE WHEN s.name IN ({",".join("?" for _ in terms[:6])}) THEN 0 ELSE 1 END,
            s.name
        LIMIT ?
        """,
        (*params, *terms[:6], limit),
    ).fetchall()
    return [
        SearchResult(
            chunk_id=row["id"],
            path=row["path"],
            title=row["title"] or row["name"],
            corpus_type=row["corpus_type"],
            topic=row["topic"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            score=0.0,
            source="symbol",
            snippet=f"{row['kind']} {row['name']} line {row['line']}: {row['signature']}",
        )
        for row in rows
    ]


def extract_identifier_terms(query: str) -> list[str]:
    normalized = query.replace("::", " ").replace("(", " ").replace(")", " ")
    terms = []
    for raw in normalized.split():
        term = raw.strip('"').strip(",.;:")
        if len(term) >= 2 and any(char.isalpha() for char in term):
            terms.append(term)
    return terms


def strong_symbol_terms(query: str) -> list[str]:
    stop_words = {
        "api",
        "asc",
        "ascend",
        "ascendc",
        "cmake",
        "npu",
        "arch",
        "simd",
        "simt",
        "cpp",
        "hello",
        "world",
    }
    terms = []
    for term in extract_identifier_terms(query):
        if not term.isascii() or len(term) < 4:
            continue
        if term.lower().replace("-", "") in stop_words:
            continue
        if "_" in term or any(char.isupper() for char in term[1:]) or term[0].isupper():
            terms.append(term)
    return terms


def _vector_search(
    conn,
    query: str,
    limit: int,
    type_filter: str | None,
    embedding_provider: str,
) -> list[SearchResult]:
    if embedding_provider == "auto":
        row = conn.execute(
            "SELECT value FROM corpus_meta WHERE key = 'embedder'"
        ).fetchone()
        embedding_provider = row["value"] if row else "auto"
    embedder = make_embedder(embedding_provider)
    query_vector = embedder.embed(query)
    params: list[object] = []
    where = ""
    if type_filter:
        where = "WHERE c.corpus_type = ?"
        params.append(type_filter)
    rows = conn.execute(
        f"""
        SELECT c.id, c.path, c.title, c.corpus_type, c.topic, c.start_line, c.end_line,
               substr(c.content, 1, 500) AS snippet, e.vector
        FROM chunks c
        JOIN embeddings e ON e.chunk_id = c.id
        {where}
        """,
        params,
    ).fetchall()
    scored: list[tuple[float, object]] = []
    for row in rows:
        score = cosine_similarity(query_vector, blob_to_vector(row["vector"]))
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        SearchResult(
            chunk_id=row["id"],
            path=row["path"],
            title=row["title"],
            corpus_type=row["corpus_type"],
            topic=row["topic"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            score=0.0,
            source="vector",
            snippet=row["snippet"] or "",
        )
        for _, row in scored[:limit]
    ]


def to_fts_query(query: str) -> str:
    terms = []
    for raw in query.replace("(", " ").replace(")", " ").replace(":", " ").split():
        term = raw.strip('"')
        if term:
            terms.append(f'"{term}"')
    return " OR ".join(terms) if terms else '""'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the Ascend C SQLite RAG index.")
    parser.add_argument("query")
    parser.add_argument("--db", type=Path, default=Path("hw_rag.sqlite"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--type", dest="type_filter")
    parser.add_argument("--embedding-provider", choices=["auto", "hash", "openai"], default="auto")
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    results = hybrid_search(
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
