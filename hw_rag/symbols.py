from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str
    signature: str
    line: int


SYMBOL_PATTERNS = [
    ("class", re.compile(r"^\s*class\s+([A-Za-z_]\w*)\b")),
    ("struct", re.compile(r"^\s*struct\s+([A-Za-z_]\w*)\b")),
    ("enum", re.compile(r"^\s*enum(?:\s+class)?\s+([A-Za-z_]\w*)\b")),
    ("macro", re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\b(.*)$")),
    (
        "function",
        re.compile(
            r"^\s*(?:template\s*<.*>\s*)?"
            r"(?:__aicore__\s+|__host__\s+|inline\s+|static\s+|constexpr\s+|typename\s+|"
            r"const\s+|virtual\s+)*"
            r"([A-Za-z_][\w:<>~*&\s]+?)\s+([A-Za-z_][\w:~]*)\s*\(([^;{}]*)\)"
        ),
    ),
]


def extract_symbols(text: str) -> list[Symbol]:
    symbols: list[Symbol] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("*"):
            continue
        for kind, pattern in SYMBOL_PATTERNS:
            match = pattern.match(line)
            if not match:
                continue
            if kind == "function":
                name = match.group(2)
            else:
                name = match.group(1)
            symbols.append(Symbol(name=name, kind=kind, signature=stripped[:500], line=line_number))
            break
    return symbols

