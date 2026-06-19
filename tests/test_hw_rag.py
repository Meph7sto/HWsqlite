from __future__ import annotations

from pathlib import Path

from hw_rag.build import build_database
from hw_rag.classifier import classify_path
from hw_rag.embeddings import HashingEmbedder
from hw_rag.search import hybrid_search


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
