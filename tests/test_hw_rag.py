from __future__ import annotations

from pathlib import Path

from hw_rag.build import build_database
from hw_rag.classifier import classify_path
from hw_rag.embeddings import HashingEmbedder
from hw_rag.search import (
    SearchResult,
    extract_identifier_terms,
    hybrid_search,
    normalize_query,
    strong_symbol_terms,
)


def test_classifies_asc_devkit_paths() -> None:
    assert classify_path("include/basic_api/kernel_tensor.h")[:2] == (
        "headers",
        "basic_api",
    )
    assert classify_path("impl/basic_api/dav_3510/kernel_operator_impl.h")[:2] == (
        "implementation",
        "basic_api",
    )
    assert classify_path("docs/api/context/DataCopy.md")[:2] == (
        "api_reference",
        "api",
    )
    assert classify_path("docs/guide/编程指南/硬件实现/硬件约束/NPU架构版本220x.md")[:2] == (
        "guide",
        "hardware_constraint",
    )
    assert classify_path("examples/01_simd_cpp_api/02_features/foo/README.md")[:2] == (
        "examples",
        "01_simd_cpp_api/02_features",
    )


def test_builds_sqlite_index_and_hybrid_searches(tmp_path: Path) -> None:
    source = tmp_path / "asc-devkit"
    source.mkdir()
    (source / "include" / "basic_api").mkdir(parents=True)
    (source / "docs" / "api" / "context").mkdir(parents=True)
    (source / "examples" / "01_simd_cpp_api" / "00_introduction").mkdir(parents=True)

    (source / "include" / "basic_api" / "kernel_operator_data_copy_intf.h").write_text(
        """
        namespace AscendC {
        template <typename T>
        __aicore__ inline void DataCopy(LocalTensor<T>& dst, GlobalTensor<T>& src, uint32_t count);
        }
        """,
        encoding="utf-8",
    )
    (source / "docs" / "api" / "context" / "DataCopy.md").write_text(
        "# DataCopy\n\n用于 GM 和 LocalTensor 之间的数据搬运。\n",
        encoding="utf-8",
    )
    (source / "examples" / "01_simd_cpp_api" / "00_introduction" / "README.md").write_text(
        "# basic_api_tque_add\n\n示例使用 TPipe、TQue、DataCopy 完成 add 算子。\n",
        encoding="utf-8",
    )

    db_path = tmp_path / "rag.sqlite"
    build_database(source, db_path, HashingEmbedder(dims=64))

    results = hybrid_search(db_path, "DataCopy LocalTensor 数据搬运", limit=5)
    assert results
    paths = {result.path for result in results}
    assert "docs/api/context/DataCopy.md" in paths
    assert any("kernel_operator_data_copy_intf.h" in path for path in paths)


def test_normalize_query_extracts_api_intent_tokens() -> None:
    normalized = normalize_query(
        "aclnnSinc custom operator docs api header prototype parameter description"
    )

    assert "sinc" in normalized.expanded_terms
    assert "prototype" not in normalized.fts_terms
    assert "parameter" not in normalized.fts_terms
    assert "description" not in normalized.fts_terms


def test_identifier_and_symbol_extraction_prefers_strong_api_terms() -> None:
    query = "aclnnop/aclnn_softmax.h aclnnSoftmaxGetWorkspaceSize aclnnSoftmax"
    normalized = normalize_query(query)

    assert normalized.primary_terms[0] == "softmax"
    assert "aclnnSoftmax" in extract_identifier_terms(query)
    assert "aclnnSoftmaxGetWorkspaceSize" in strong_symbol_terms(query)


def test_hybrid_search_skips_vector_for_exact_api_doc_hits(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "asc-devkit"
    source.mkdir()
    (source / "docs" / "api" / "context").mkdir(parents=True)
    (source / "include" / "activation").mkdir(parents=True)

    (source / "docs" / "api" / "context" / "SoftMax.md").write_text(
        "# SoftMax\n\n函数原型\n\n参数说明\n",
        encoding="utf-8",
    )
    (source / "include" / "activation" / "softmax.h").write_text(
        "inline void SoftMax(float* dst, const float* src, int count);\n",
        encoding="utf-8",
    )

    db_path = tmp_path / "rag.sqlite"
    build_database(source, db_path, HashingEmbedder(dims=64))

    called = False

    def fail_vector(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        raise AssertionError("vector search should not run when exact doc hits are enough")

    monkeypatch.setattr("hw_rag.search._vector_search", fail_vector)

    results = hybrid_search(db_path, "softmax", limit=5, type_filter="api_reference")

    assert results
    assert results[0].path == "docs/api/context/SoftMax.md"
    assert called is False


def test_hybrid_search_reranks_narrowed_candidates(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "asc-devkit"
    source.mkdir()
    (source / "docs" / "api" / "context").mkdir(parents=True)
    (source / "include" / "activation").mkdir(parents=True)

    (source / "docs" / "api" / "context" / "SoftMax.md").write_text(
        "# SoftMax\n\n对输入tensor按行做Softmax计算。\n",
        encoding="utf-8",
    )
    (source / "docs" / "api" / "context" / "LogSoftMax.md").write_text(
        "# LogSoftMax\n\n对输入tensor做LogSoftmax计算。\n",
        encoding="utf-8",
    )
    (source / "include" / "activation" / "softmax_impl.h").write_text(
        "inline void SoftMaxImpl(float* dst, const float* src, int count);\n",
        encoding="utf-8",
    )

    db_path = tmp_path / "rag.sqlite"
    build_database(source, db_path, HashingEmbedder(dims=64))

    captured_candidates: list[int] = []

    def fake_vector_search(
        conn,
        query: str,
        limit: int,
        type_filter: str | None,
        embedding_provider: str,
        candidate_ids: set[int] | None = None,
    ) -> list[SearchResult]:
        assert candidate_ids is not None
        captured_candidates.extend(sorted(candidate_ids))
        rows = conn.execute(
            "SELECT id, path, title, corpus_type, topic, start_line, end_line FROM chunks WHERE id IN ({})".format(
                ",".join("?" for _ in candidate_ids)
            ),
            tuple(candidate_ids),
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
                source="vector",
                snippet="",
            )
            for row in rows
            if "SoftMax" in row["path"] or "softmax" in row["path"].lower()
        ][:limit]

    monkeypatch.setattr("hw_rag.search._vector_search", fake_vector_search)

    results = hybrid_search(db_path, "aclnnSoftmaxGetWorkspaceSize aclnnSoftmax softmax", limit=5)

    assert results
    assert captured_candidates
    assert len(captured_candidates) < 10
    assert any(result.path == "docs/api/context/SoftMax.md" for result in results)
