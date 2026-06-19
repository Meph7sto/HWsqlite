from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


LEGACY_DEFAULT_DB = Path("hw_rag.sqlite")
DEFAULT_MANIFEST = Path("hw_rag.dataset.json")


@dataclass(frozen=True)
class SourceConfig:
    name: str
    kind: str
    root: Path
    include_tools: bool = True
    path_prefix: str | None = None

    def resolved(self) -> SourceConfig:
        return SourceConfig(
            name=self.name,
            kind=self.kind,
            root=self.root.resolve(),
            include_tools=self.include_tools,
            path_prefix=self.path_prefix,
        )

    @property
    def storage_prefix(self) -> str:
        if self.path_prefix is None:
            return self.name.strip("/")
        return self.path_prefix.strip("/")

    def storage_path_for(self, relative_path: str) -> str:
        normalized = relative_path.replace("\\", "/")
        if not self.storage_prefix:
            return normalized
        return f"{self.storage_prefix}/{normalized}"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    sources: tuple[SourceConfig, ...]
    db_path: Path | None = None

    def resolved(self) -> DatasetConfig:
        return DatasetConfig(
            name=self.name,
            sources=tuple(source.resolved() for source in self.sources),
            db_path=self.db_path.resolve() if self.db_path is not None else None,
        )


def load_dataset_manifest(manifest_path: Path) -> DatasetConfig:
    manifest_path = manifest_path.resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    sources = tuple(_load_source_config(item, base_dir) for item in data["sources"])
    db_raw = data.get("db_path")
    db_path = (base_dir / db_raw).resolve() if db_raw else None
    return DatasetConfig(
        name=str(data["name"]),
        sources=sources,
        db_path=db_path,
    )


def default_manifest_path(base_dir: Path | None = None) -> Path:
    root = base_dir.resolve() if base_dir is not None else Path.cwd().resolve()
    return root / DEFAULT_MANIFEST


def default_dataset(base_dir: Path | None = None) -> DatasetConfig | None:
    manifest_path = default_manifest_path(base_dir)
    if not manifest_path.exists():
        return None
    return load_dataset_manifest(manifest_path)


def default_db_path(base_dir: Path | None = None) -> Path:
    dataset = default_dataset(base_dir)
    if dataset is not None and dataset.db_path is not None:
        return dataset.db_path
    root = base_dir.resolve() if base_dir is not None else Path.cwd().resolve()
    return root / LEGACY_DEFAULT_DB


def detect_source_kind(source_root: Path) -> str:
    root = source_root.resolve()
    if (root / "include").is_dir() and (root / "docs").is_dir():
        return "asc-devkit"
    if any((root / name).is_dir() for name in ("ops-nn", "ops-math", "ops-cv", "ops-transformer")):
        return "cann-docs"
    return "generic-docs"


def single_source_dataset(
    source_root: Path,
    *,
    name: str | None = None,
    kind: str | None = None,
    include_tools: bool = True,
    path_prefix: str = "",
    dataset_name: str = "single-source",
) -> DatasetConfig:
    root = source_root.resolve()
    resolved_kind = kind or detect_source_kind(root)
    resolved_name = name or root.name
    return DatasetConfig(
        name=dataset_name,
        sources=(
            SourceConfig(
                name=resolved_name,
                kind=resolved_kind,
                root=root,
                include_tools=include_tools,
                path_prefix=path_prefix,
            ),
        ),
    )


def dataset_to_metadata(dataset: DatasetConfig) -> dict[str, object]:
    return {
        "name": dataset.name,
        "db_path": str(dataset.db_path) if dataset.db_path is not None else None,
        "sources": [
            {
                "name": source.name,
                "kind": source.kind,
                "root": str(source.root),
                "include_tools": source.include_tools,
                "path_prefix": source.path_prefix,
            }
            for source in dataset.sources
        ],
    }


def _load_source_config(item: dict[str, object], base_dir: Path) -> SourceConfig:
    root = (base_dir / str(item["root"])).resolve()
    return SourceConfig(
        name=str(item["name"]),
        kind=str(item.get("kind") or detect_source_kind(root)),
        root=root,
        include_tools=bool(item.get("include_tools", True)),
        path_prefix=str(item.get("path_prefix", item["name"])),
    )
