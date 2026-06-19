from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


MAX_CHARS = 6000
OVERLAP_CHARS = 600


@dataclass(frozen=True)
class Chunk:
    title: str
    content: str
    start_line: int
    end_line: int


def chunk_text(relative_path: str, text: str) -> list[Chunk]:
    suffix = Path(relative_path).suffix.lower()
    if suffix == ".md":
        return chunk_markdown(text)
    if suffix in {".h", ".hpp", ".c", ".cc", ".cpp", ".asc", ".aicpu"}:
        return chunk_code(text)
    return chunk_by_size(text, title=Path(relative_path).name)


def chunk_markdown(text: str) -> list[Chunk]:
    lines = text.splitlines()
    heading_indices = [
        index for index, line in enumerate(lines) if re.match(r"^#{1,4}\s+", line)
    ]
    if not heading_indices:
        return chunk_by_size(text, title="")

    chunks: list[Chunk] = []
    for position, start in enumerate(heading_indices):
        end = heading_indices[position + 1] if position + 1 < len(heading_indices) else len(lines)
        title = re.sub(r"<a name=.*?</a>", "", lines[start]).lstrip("#").strip()
        section = "\n".join(lines[start:end]).strip()
        for split in split_large_chunk(section, title, start + 1):
            chunks.append(split)
    return chunks


def chunk_code(text: str) -> list[Chunk]:
    lines = text.splitlines()
    chunks: list[Chunk] = []
    current: list[str] = []
    start_line = 1
    title = ""

    def flush(end_line: int) -> None:
        nonlocal current, start_line, title
        content = "\n".join(current).strip()
        if content:
            chunks.extend(split_large_chunk(content, title, start_line))
        current = []
        title = ""

    pattern = re.compile(
        r"^\s*(template\s*<.*>\s*)?"
        r"((class|struct|enum)\s+\w+|"
        r"([A-Za-z_][\w:<>~*&\s]+)\s+[A-Za-z_][\w:~]*\s*\([^;]*\)\s*(const)?\s*(\{|;)?|"
        r"#\s*define\s+\w+)"
    )

    for line_number, line in enumerate(lines, start=1):
        if pattern.match(line) and current and len("\n".join(current)) > 1200:
            flush(line_number - 1)
            start_line = line_number
        if not current:
            start_line = line_number
        if not title and pattern.match(line):
            title = line.strip()[:180]
        current.append(line)
        if len("\n".join(current)) >= MAX_CHARS:
            flush(line_number)
    flush(len(lines))
    return chunks


def chunk_by_size(text: str, title: str, base_line: int = 1) -> list[Chunk]:
    lines = text.splitlines()
    chunks: list[Chunk] = []
    current: list[str] = []
    start_line = base_line
    char_count = 0
    for offset, line in enumerate(lines):
        line_number = base_line + offset
        if not current:
            start_line = line_number
        current.append(line)
        char_count += len(line) + 1
        if char_count >= MAX_CHARS:
            chunks.append(Chunk(title, "\n".join(current).strip(), start_line, line_number))
            overlap = "\n".join(current)[-OVERLAP_CHARS:].splitlines()
            current = overlap
            start_line = max(base_line, line_number - len(overlap) + 1)
            char_count = sum(len(item) + 1 for item in current)
    if current:
        chunks.append(Chunk(title, "\n".join(current).strip(), start_line, base_line + len(lines) - 1))
    return [chunk for chunk in chunks if chunk.content]


def split_large_chunk(content: str, title: str, start_line: int) -> list[Chunk]:
    if len(content) <= MAX_CHARS:
        line_count = content.count("\n") + 1
        return [Chunk(title, content, start_line, start_line + line_count - 1)]
    return chunk_by_size(content, title, base_line=start_line)
