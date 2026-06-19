from __future__ import annotations

import json
from pathlib import Path

from hw_rag.build import build_database, build_dataset
from hw_rag.classifier import classify_path, classify_source_path
from hw_rag.corpus import DatasetConfig, SourceConfig, load_dataset_manifest
from hw_rag.db import connect
from hw_rag.embeddings import HashingEmbedder
from hw_rag.mcp_server import validate_db_path
from hw_rag.plugins import (
    SourcePlugin,
    get_plugin,
    list_plugins,
    register_plugin,
    source_plugin,
)
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


def test_classifies_cann_docs_paths() -> None:
    assert classify_source_path("cann-docs", "ops-nn/aclnnAddRmsNorm.md")[:2] == (
        "api_reference",
        "nn",
    )
    assert classify_source_path("cann-docs", "ops-transformer/op_api_list.md")[:2] == (
        "api_catalog",
        "transformer",
    )
    assert classify_source_path("cann-docs", "common/数据类型.md")[:2] == (
        "guide",
        "common",
    )


def test_builtin_source_plugins_are_registered() -> None:
    plugin_names = {plugin.kind for plugin in list_plugins()}

    assert "asc-devkit" in plugin_names
    assert "cann-docs" in plugin_names
    assert "generic-docs" in plugin_names
    assert get_plugin("asc-devkit").classify_path("docs/api/context/DataCopy.md")[:2] == (
        "api_reference",
        "api",
    )


def test_can_register_runtime_source_plugin(tmp_path: Path) -> None:
    @source_plugin("demo-plugin")
    class DemoPlugin(SourcePlugin):
        def classify_path(self, relative_path: str) -> tuple[str, str, str]:
            return ("demo_docs", "demo", "Demo docs")

        def should_skip(self, relative_path: str, path: Path, include_tools: bool) -> bool:
            return relative_path.endswith(".skip")

    plugin = get_plugin("demo-plugin")

    assert plugin.classify_path("anything.txt") == ("demo_docs", "demo", "Demo docs")
    assert plugin.should_skip("ignore.skip", tmp_path / "ignore.skip", include_tools=True) is True


def test_loads_dataset_manifest_and_resolves_paths(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "config"
    manifest_dir.mkdir()
    asc_root = tmp_path / "vendor" / "asc-devkit-relocated"
    cann_root = tmp_path / "mirror" / "cann-docs"
    asc_root.mkdir(parents=True)
    cann_root.mkdir(parents=True)

    manifest_path = manifest_dir / "hybrid.dataset.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "hybrid-test",
                "db_path": "../build/hybrid.sqlite",
                "sources": [
                    {
                        "name": "asc-core",
                        "kind": "asc-devkit",
                        "root": "../vendor/asc-devkit-relocated",
                    },
                    {
                        "name": "cann-docs",
                        "kind": "cann-docs",
                        "root": "../mirror/cann-docs",
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    dataset = load_dataset_manifest(manifest_path)

    assert dataset.name == "hybrid-test"
    assert dataset.db_path == (tmp_path / "build" / "hybrid.sqlite").resolve()
    assert dataset.sources[0].root == asc_root.resolve()
    assert dataset.sources[1].root == cann_root.resolve()


def test_build_pipeline_dispatches_through_registered_plugin(tmp_path: Path) -> None:
    @register_plugin
    class DemoBuildPlugin(SourcePlugin):
        kind = "demo-build"

        def classify_path(self, relative_path: str) -> tuple[str, str, str]:
            return ("demo_reference", "demo", "Demo plugin content")

        def should_skip(self, relative_path: str, path: Path, include_tools: bool) -> bool:
            return relative_path.endswith(".skip")

    source_root = tmp_path / "demo-source"
    source_root.mkdir()
    (source_root / "keep.txt").write_text("demo text", encoding="utf-8")
    (source_root / "drop.skip").write_text("skip me", encoding="utf-8")

    dataset = DatasetConfig(
        name="demo-dataset",
        sources=(SourceConfig(name="demo", kind="demo-build", root=source_root),),
    )
    db_path = tmp_path / "demo.sqlite"
    build_dataset(dataset, db_path, HashingEmbedder(dims=32))

    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT path, corpus_type, topic, role FROM documents"
        ).fetchone()
        count = conn.execute("SELECT COUNT(*) AS count FROM documents").fetchone()["count"]

    assert count == 1
    assert row["path"] == "demo/keep.txt"
    assert row["corpus_type"] == "demo_reference"
    assert row["topic"] == "demo"


def test_validate_db_path_rejects_missing_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.sqlite"
    db_path.touch()

    try:
        validate_db_path(db_path)
    except ValueError as exc:
        assert "missing required tables" in str(exc)
    else:
        raise AssertionError("validate_db_path should reject an empty sqlite file")


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


def test_builds_multi_source_dataset_and_searches(tmp_path: Path) -> None:
    asc_root = tmp_path / "vendor" / "asc-devkit-relocated"
    cann_root = tmp_path / "docs" / "cann-docs"

    (asc_root / "docs" / "api" / "context").mkdir(parents=True)
    (asc_root / "include" / "basic_api").mkdir(parents=True)
    (cann_root / "ops-nn").mkdir(parents=True)
    (cann_root / "common").mkdir(parents=True)

    (asc_root / "docs" / "api" / "context" / "DataCopy.md").write_text(
        "# DataCopy\n\n用于 GM 和 LocalTensor 之间的数据搬运。\n",
        encoding="utf-8",
    )
    (asc_root / "include" / "basic_api" / "kernel_operator_data_copy_intf.h").write_text(
        """
        namespace AscendC {
        template <typename T>
        __aicore__ inline void DataCopy(LocalTensor<T>& dst, GlobalTensor<T>& src, uint32_t count);
        }
        """,
        encoding="utf-8",
    )
    (cann_root / "ops-nn" / "aclnnAddRmsNorm.md").write_text(
        "# aclnnAddRmsNorm\n\n## 接口原型\n\naclnnAddRmsNorm(...)\n",
        encoding="utf-8",
    )
    (cann_root / "common" / "数据类型.md").write_text(
        "# 数据类型\n\n介绍 CANN 文档中的常见数据类型。\n",
        encoding="utf-8",
    )

    dataset = DatasetConfig(
        name="hybrid-docs",
        sources=(
            SourceConfig(name="asc-devkit", kind="asc-devkit", root=asc_root),
            SourceConfig(name="cann-docs", kind="cann-docs", root=cann_root),
        ),
    )
    db_path = tmp_path / "hybrid.sqlite"
    build_dataset(dataset, db_path, HashingEmbedder(dims=64))

    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT source_name, relative_path, path FROM documents ORDER BY source_name, relative_path"
        ).fetchall()
        assert [row["source_name"] for row in rows] == [
            "asc-devkit",
            "asc-devkit",
            "cann-docs",
            "cann-docs",
        ]
        assert rows[0]["path"].startswith("asc-devkit/")
        assert rows[-1]["path"].startswith("cann-docs/")

    asc_results = hybrid_search(db_path, "DataCopy LocalTensor 数据搬运", limit=5)
    assert any(
        result.source_name == "asc-devkit"
        and result.path == "asc-devkit/docs/api/context/DataCopy.md"
        for result in asc_results
    )

    cann_results = hybrid_search(db_path, "aclnnAddRmsNorm", limit=5)
    assert cann_results
    assert cann_results[0].source_name == "cann-docs"
    assert cann_results[0].path == "cann-docs/ops-nn/aclnnAddRmsNorm.md"


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
                source_name="asc-devkit",
                source_kind="asc-devkit",
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
