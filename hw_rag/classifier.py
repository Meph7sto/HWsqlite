from __future__ import annotations

from pathlib import Path

from .plugins import CODE_EXTENSIONS, TEXT_EXTENSIONS, get_plugin


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS


def classify_path(relative_path: str) -> tuple[str, str, str]:
    return classify_source_path("asc-devkit", relative_path)


def classify_source_path(source_kind: str, relative_path: str) -> tuple[str, str, str]:
    return get_plugin(source_kind).classify_path(relative_path)
