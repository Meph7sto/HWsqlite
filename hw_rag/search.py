from __future__ import annotations

import argparse
import heapq
import math
import re
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


GENERIC_QUERY_TERMS = {
    "api",
    "apis",
    "ascend",
    "cann",
    "context",
    "custom",
    "description",
    "doc",
    "docs",
    "documentation",
    "header",
    "headers",
    "official",
    "operator",
    "operators",
    "parameter",
    "parameters",
    "prototype",
}

VECTOR_AUXILIARY_TERMS = {
    "buffer",
    "example",
    "examples",
    "implementation",
    "tiling",
    "tmp",
    "workspace",
}

SYMBOL_STOP_WORDS = {
    "api",
    "apis",
    "asc",
    "ascend",
    "ascendc",
    "cann",
    "cmake",
    "cpp",
    "header",
    "headers",
    "hello",
    "implementation",
    "npu",
    "official",
    "prototype",
    "simd",
    "simt",
    "world",
}

API_PREFIXES = ("aclnn", "aclnnop", "aclop")
API_SUFFIX_PATTERNS = (
    ("get", "workspace", "size"),
    ("workspace", "size"),
    ("get", "workspace"),
    ("get", "tmp", "size"),
    ("tmp", "size"),
    ("get", "tiling", "key"),
    ("tiling", "key"),
)
EARLY_EXIT_CORPORA = {"api_reference", "headers", "implementation"}


@dataclass(frozen=True)
class NormalizedQuery:
    raw: str
    identifier_terms: tuple[str, ...]
    expanded_terms: tuple[str, ...]
    primary_terms: tuple[str, ...]
    exact_terms: tuple[str, ...]
    fts_terms: tuple[str, ...]
    symbol_terms: tuple[str, ...]
    semantic_terms: tuple[str, ...]

    @property
    def semantic_query(self) -> str:
        if self.semantic_terms:
            return " ".join(self.semantic_terms)
        if self.fts_terms:
            return " ".join(self.fts_terms)
        return self.raw


def hybrid_search(
    db_path: Path,
    query: str,
    limit: int = 10,
    type_filter: str | None = None,
    embedding_provider: str = "auto",
) -> list[SearchResult]:
    normalized = normalize_query(query)
    with connect(db_path) as conn:
        metadata = _metadata_search(conn, normalized, max(limit * 4, 12), type_filter)
        fts = _fts_search(conn, normalized, limit * 4, type_filter)
        symbols = (
            []
            if should_skip_symbol_search(normalized, type_filter)
            else _symbol_search(conn, normalized, limit * 3, type_filter)
        )
        vector: list[SearchResult] = []
        if not should_skip_vector_search(metadata, normalized, type_filter):
            candidate_ids = collect_candidate_ids(
                metadata,
                fts,
                symbols,
                max_candidates=limit * 16,
            )
            vector = _vector_search(
                conn,
                normalized.semantic_query,
                limit * 4,
                type_filter,
                embedding_provider,
                candidate_ids=candidate_ids or None,
            )

    merged: dict[int, SearchResult] = {}
    for weight, source_results in (
        (2.2, metadata),
        (1.0, fts),
        (1.25, symbols),
        (1.1, vector),
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


def normalize_query(query: str) -> NormalizedQuery:
    identifier_terms = tuple(extract_identifier_terms(query))
    expanded_terms = unique_preserving_order(
        variant
        for term in identifier_terms
        for variant in expand_query_variants(term)
    )
    primary_terms = unique_preserving_order(
        primary
        for term in identifier_terms
        for primary in [extract_primary_term(term)]
        if primary and primary not in GENERIC_QUERY_TERMS
    )
    fts_terms = unique_preserving_order(
        term
        for term in [*primary_terms, *expanded_terms]
        if is_useful_fts_term(term)
    )
    symbol_terms = tuple(strong_symbol_terms(query))
    semantic_terms = unique_preserving_order(
        term
        for term in [*primary_terms, *fts_terms]
        if term not in GENERIC_QUERY_TERMS or term in VECTOR_AUXILIARY_TERMS
    )
    exact_terms = unique_preserving_order(
        term
        for term in [*primary_terms, *expanded_terms]
        if is_useful_exact_term(term)
    )
    return NormalizedQuery(
        raw=query,
        identifier_terms=identifier_terms,
        expanded_terms=tuple(expanded_terms),
        primary_terms=tuple(primary_terms),
        exact_terms=tuple(exact_terms),
        fts_terms=tuple(fts_terms),
        symbol_terms=symbol_terms,
        semantic_terms=tuple(semantic_terms),
    )


def unique_preserving_order(items) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def extract_primary_term(term: str) -> str | None:
    words = split_query_words(term)
    if not words:
        return None
    lowered_words = [word.lower() for word in words]
    without_prefix = strip_redundant_api_prefix_words(lowered_words)
    lowered = strip_api_suffix_words(without_prefix)
    if not lowered:
        return None
    primary = "".join(lowered)
    return primary if len(primary) >= 3 else None


def expand_query_variants(term: str) -> list[str]:
    cleaned = term.strip('"').strip(",.;:")
    if not cleaned:
        return []

    variants: list[str] = []
    if "/" not in cleaned and "\\" not in cleaned and "." not in cleaned:
        variants.append(cleaned.lower())

    basename = Path(cleaned.replace("\\", "/")).name
    stem = Path(basename).stem
    if stem:
        variants.append(stem.lower())

    words = [word.lower() for word in split_query_words(cleaned)]
    if words:
        variants.append("".join(words))
        if len(words) > 1:
            variants.append("_".join(words))

    primary = extract_primary_term(cleaned)
    if primary:
        variants.append(primary)

    return unique_preserving_order(variants)


def split_query_words(term: str) -> list[str]:
    normalized = term.replace("\\", "/")
    parts = [Path(piece).stem for piece in normalized.split("/") if piece]
    words: list[str] = []
    for part in parts:
        for piece in re.split(r"[_\-.]+", part):
            if not piece:
                continue
            matches = re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+|[\u4e00-\u9fff]+", piece)
            words.extend(matches or [piece])
    return words


def strip_api_prefix_words(words: list[str]) -> list[str]:
    if words and words[0] in API_PREFIXES:
        return words[1:]
    return words


def strip_redundant_api_prefix_words(words: list[str]) -> list[str]:
    stripped = list(words)
    while stripped and stripped[0] in API_PREFIXES:
        stripped = stripped[1:]
    return stripped


def strip_api_suffix_words(words: list[str]) -> list[str]:
    for pattern in API_SUFFIX_PATTERNS:
        if len(words) >= len(pattern) and tuple(words[-len(pattern):]) == pattern:
            return words[:-len(pattern)]
    return words


def is_useful_fts_term(term: str) -> bool:
    return (
        len(term) >= 3
        and (term not in GENERIC_QUERY_TERMS or term in VECTOR_AUXILIARY_TERMS)
    )


def is_useful_exact_term(term: str) -> bool:
    return len(term) >= 3 and term not in GENERIC_QUERY_TERMS and term != "size"


def _fts_search(
    conn,
    normalized: NormalizedQuery,
    limit: int,
    type_filter: str | None,
) -> list[SearchResult]:
    terms = normalized.fts_terms or normalized.exact_terms or (normalized.raw,)
    where = "chunk_fts MATCH ?"
    params: list[object] = [to_fts_query(terms)]
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


def _metadata_search(
    conn,
    normalized: NormalizedQuery,
    limit: int,
    type_filter: str | None,
) -> list[SearchResult]:
    terms = normalized.exact_terms[:8]
    if not terms:
        return []

    score_clauses: list[str] = []
    score_params: list[object] = []
    where_clauses: list[str] = []
    where_params: list[object] = []
    for term in terms:
        score_clauses.extend(
            [
                "CASE WHEN lower(c.title) = ? THEN 120 ELSE 0 END",
                "CASE WHEN lower(c.path) LIKE ? THEN 110 ELSE 0 END",
                "CASE WHEN lower(c.path) LIKE ? THEN 105 ELSE 0 END",
                "CASE WHEN lower(c.path) LIKE ? THEN 100 ELSE 0 END",
                "CASE WHEN instr(lower(c.title), ?) > 0 THEN 45 ELSE 0 END",
                "CASE WHEN instr(lower(c.path), ?) > 0 THEN 35 ELSE 0 END",
            ]
        )
        score_params.extend(
            [
                term,
                f"%/{term}.md",
                f"%/{term}.h",
                f"%/{term}.hpp",
                term,
                term,
            ]
        )
        where_clauses.extend(
            [
                "lower(c.title) = ?",
                "lower(c.path) LIKE ?",
                "lower(c.path) LIKE ?",
                "lower(c.path) LIKE ?",
                "instr(lower(c.title), ?) > 0",
                "instr(lower(c.path), ?) > 0",
            ]
        )
        where_params.extend(
            [
                term,
                f"%/{term}.md",
                f"%/{term}.h",
                f"%/{term}.hpp",
                term,
                term,
            ]
        )

    where = "(" + " OR ".join(where_clauses) + ")"
    params: list[object] = [*score_params, *where_params]
    if type_filter:
        where += " AND c.corpus_type = ?"
        params.append(type_filter)

    rows = conn.execute(
        f"""
        SELECT c.id, c.path, c.title, c.corpus_type, c.topic, c.start_line, c.end_line,
               substr(c.content, 1, 500) AS snippet,
               {" + ".join(score_clauses)} AS match_score
        FROM chunks c
        WHERE {where}
        ORDER BY
            match_score DESC,
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
            source="metadata",
            snippet=row["snippet"] or "",
        )
        for row in rows
    ]


def should_skip_symbol_search(normalized: NormalizedQuery, type_filter: str | None) -> bool:
    if type_filter == "api_reference":
        return True
    return not normalized.symbol_terms


def _symbol_search(
    conn,
    normalized: NormalizedQuery,
    limit: int,
    type_filter: str | None,
) -> list[SearchResult]:
    terms = normalized.symbol_terms
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
    terms: list[str] = []
    for term in extract_identifier_terms(query):
        if not term.isascii() or len(term) < 4:
            continue
        normalized = term.lower().replace("-", "")
        if normalized in SYMBOL_STOP_WORDS:
            continue
        if "_" in term or any(char.isupper() for char in term[1:]) or term[0].isupper():
            terms.append(term)
            stem = Path(term.replace("\\", "/")).stem
            if stem != term and stem.lower() not in SYMBOL_STOP_WORDS:
                terms.append(stem)
            primary = extract_primary_term(term)
            if (
                primary
                and primary != normalized
                and ("_" in term or "/" in term or "\\" in term or primary != normalized.lstrip("_"))
                and not normalized.startswith(("aclnn", "aclop"))
            ):
                terms.append(primary)
    return unique_preserving_order(terms)


def _vector_search(
    conn,
    query: str,
    limit: int,
    type_filter: str | None,
    embedding_provider: str,
    candidate_ids: set[int] | None = None,
) -> list[SearchResult]:
    if embedding_provider == "auto":
        row = conn.execute(
            "SELECT value FROM corpus_meta WHERE key = 'embedder'"
        ).fetchone()
        embedding_provider = row["value"] if row else "auto"
    embedder = make_embedder(embedding_provider)
    query_vector = embedder.embed(query)
    query_norm = math.sqrt(sum(value * value for value in query_vector))
    if query_norm == 0:
        return []
    params: list[object] = []
    clauses: list[str] = []
    if type_filter:
        clauses.append("c.corpus_type = ?")
        params.append(type_filter)
    if candidate_ids:
        placeholders = ",".join("?" for _ in candidate_ids)
        clauses.append(f"c.id IN ({placeholders})")
        params.extend(sorted(candidate_ids))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT c.id, c.path, c.title, c.corpus_type, c.topic, c.start_line, c.end_line,
               substr(c.content, 1, 500) AS snippet, e.vector
        FROM chunks c
        JOIN embeddings e ON e.chunk_id = c.id
        {where}
        """,
        params,
    )
    scored: list[tuple[float, int, object]] = []
    for row in rows:
        vector = blob_to_vector(row["vector"])
        score = cosine_similarity_from_norm(query_vector, query_norm, vector)
        if score > 0:
            item = (score, int(row["id"]), row)
            if len(scored) < limit:
                heapq.heappush(scored, item)
            elif score > scored[0][0]:
                heapq.heapreplace(scored, item)
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
        for _, _, row in scored[:limit]
    ]


def collect_candidate_ids(*result_groups: list[SearchResult], max_candidates: int) -> set[int]:
    candidate_ids: list[int] = []
    seen: set[int] = set()
    for group in result_groups:
        for result in group:
            if result.chunk_id in seen:
                continue
            seen.add(result.chunk_id)
            candidate_ids.append(result.chunk_id)
            if len(candidate_ids) >= max_candidates:
                return set(candidate_ids)
    return set(candidate_ids)


def should_skip_vector_search(
    metadata: list[SearchResult],
    normalized: NormalizedQuery,
    type_filter: str | None,
) -> bool:
    if type_filter not in EARLY_EXIT_CORPORA:
        return False
    if not metadata:
        return False
    top_match = metadata[0]
    path_lower = top_match.path.lower()
    title_lower = (top_match.title or "").lower()
    if type_filter in {"headers", "implementation"} and top_match.source == "metadata":
        primary_terms = normalized.primary_terms or normalized.exact_terms
        for term in primary_terms:
            if term in path_lower:
                return True
    for term in normalized.primary_terms or normalized.exact_terms:
        if title_lower == term:
            return True
        if path_lower.endswith(f"/{term}.md") or path_lower.endswith(f"/{term}.h") or path_lower.endswith(f"/{term}.hpp"):
            return True
    return False


def cosine_similarity_from_norm(left: list[float], left_norm: float, right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    right_norm = math.sqrt(sum(value * value for value in right))
    if right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def to_fts_query(query: str | tuple[str, ...] | list[str]) -> str:
    raw_terms = [query] if isinstance(query, str) else list(query)
    terms = []
    for raw in raw_terms:
        if not isinstance(raw, str):
            continue
        for piece in raw.replace("(", " ").replace(")", " ").replace(":", " ").split():
            term = piece.strip('"')
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
