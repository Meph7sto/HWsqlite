from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk, chunk_text
from .symbols import Symbol, extract_symbols


TEXT_EXTENSIONS = {
    "",
    ".aicpu",
    ".asc",
    ".awk",
    ".bash",
    ".c",
    ".cc",
    ".cfg",
    ".cmake",
    ".cpp",
    ".csh",
    ".csv",
    ".fish",
    ".h",
    ".hpp",
    ".in",
    ".inc",
    ".info",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

CODE_EXTENSIONS = {
    ".aicpu",
    ".asc",
    ".c",
    ".cc",
    ".cmake",
    ".cpp",
    ".h",
    ".hpp",
    ".py",
    ".sh",
}


class SourcePlugin:
    kind = "generic-docs"

    def is_text_file(self, path: Path) -> bool:
        return path.suffix.lower() in TEXT_EXTENSIONS

    def classify_path(self, relative_path: str) -> tuple[str, str, str]:
        parts = Path(relative_path).parts
        top = parts[0] if parts else "root"
        suffix = Path(relative_path).suffix.lower()
        if suffix == ".md":
            return "guide", top, "Documentation content"
        if suffix in CODE_EXTENSIONS:
            return "project_code", top, "Repository source file"
        return "project_metadata", top, "Repository metadata"

    def should_skip(self, relative_path: str, path: Path, include_tools: bool) -> bool:
        normalized = relative_path.replace("\\", "/")
        if "/figures/" in f"/{normalized}" or path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            return True
        if any(part in {".git", "__pycache__", "build", "out"} for part in Path(relative_path).parts):
            return True
        if not include_tools and normalized.startswith(("tools/", "scripts/", "tests/")):
            return True
        return not self.is_text_file(path)

    def chunk_text(self, relative_path: str, text: str) -> list[Chunk]:
        return chunk_text(relative_path, text)

    def should_extract_symbols(self, path: Path) -> bool:
        return path.suffix.lower() in CODE_EXTENSIONS

    def extract_symbols(self, text: str) -> list[Symbol]:
        return extract_symbols(text)


@dataclass(frozen=True)
class RegisteredPlugin:
    kind: str
    plugin: SourcePlugin


_PLUGIN_REGISTRY: dict[str, SourcePlugin] = {}


def register_plugin(plugin_type: type[SourcePlugin]) -> type[SourcePlugin]:
    plugin = plugin_type()
    if not getattr(plugin, "kind", ""):
        raise ValueError("SourcePlugin.kind must be a non-empty string")
    _PLUGIN_REGISTRY[plugin.kind] = plugin
    return plugin_type


def source_plugin(kind: str):
    def decorator(plugin_type: type[SourcePlugin]) -> type[SourcePlugin]:
        plugin_type.kind = kind
        return register_plugin(plugin_type)

    return decorator


def get_plugin(kind: str) -> SourcePlugin:
    plugin = _PLUGIN_REGISTRY.get(kind)
    if plugin is None:
        raise KeyError(f"Unknown source plugin: {kind}")
    return plugin


def list_plugins() -> list[RegisteredPlugin]:
    return [
        RegisteredPlugin(kind=kind, plugin=plugin)
        for kind, plugin in sorted(_PLUGIN_REGISTRY.items())
    ]


@register_plugin
class GenericDocsPlugin(SourcePlugin):
    kind = "generic-docs"


@register_plugin
class AscDevkitPlugin(SourcePlugin):
    kind = "asc-devkit"

    def classify_path(self, relative_path: str) -> tuple[str, str, str]:
        parts = Path(relative_path).parts
        top = parts[0] if parts else ""
        path = relative_path.replace("\\", "/")
        suffix = Path(relative_path).suffix.lower()

        if relative_path == "README.md":
            return "project_overview", "root", "Project overview"
        if top == "include":
            return "headers", _api_family(parts), "API declarations"
        if top == "impl":
            return "implementation", _api_family(parts), "API implementations"
        if top == "docs":
            return _classify_asc_docs(path, parts)
        if top == "examples":
            return _classify_examples(parts)
        if top == "tests":
            return _classify_tests(parts)
        if top == "cmake" or relative_path in {"CMakeLists.txt", "version.cmake"}:
            return "build_system", "cmake", "CMake build modules"
        if top == "scripts" or relative_path == "build.sh":
            return "packaging_scripts", "scripts", "Packaging and release scripts"
        if top == "tools":
            return "tools", parts[1] if len(parts) > 1 else "tools", "Build/runtime tools"
        if top in {".devcontainer", ".vscode"}:
            return "dev_environment", top.lstrip("."), "Local development environment"
        if top == ".gitcode":
            return "repo_process", "gitcode", "Repository collaboration metadata"
        if suffix in CODE_EXTENSIONS:
            return "project_code", top, "Repository source file"
        return "project_metadata", top or "root", "Repository metadata"

    def should_skip(self, relative_path: str, path: Path, include_tools: bool) -> bool:
        if super().should_skip(relative_path, path, include_tools):
            return True
        normalized = relative_path.replace("\\", "/")
        return normalized.startswith("scripts/package/asc-devkit/")


@register_plugin
class CannDocsPlugin(SourcePlugin):
    kind = "cann-docs"

    def classify_path(self, relative_path: str) -> tuple[str, str, str]:
        path = relative_path.replace("\\", "/")
        parts = Path(relative_path).parts
        top = parts[0] if parts else ""
        if top == "root":
            return "docs_overview", "root", "CANN documentation root navigation"
        if top == "common":
            return _classify_cann_common(path)
        if top.startswith("ops-"):
            return _classify_cann_ops(path, top)
        suffix = Path(relative_path).suffix.lower()
        if suffix in CODE_EXTENSIONS:
            return "project_code", top or "root", "Repository source file"
        return "project_metadata", top or "root", "Repository metadata"


def _api_family(parts: tuple[str, ...]) -> str:
    if len(parts) < 2:
        return "root"
    if parts[1] == "experimental" and len(parts) >= 3:
        return f"experimental/{parts[2]}"
    return parts[1]


def _classify_asc_docs(path: str, parts: tuple[str, ...]) -> tuple[str, str, str]:
    if path.startswith("docs/api/"):
        if path == "docs/api/README.md":
            return "api_catalog", "api", "Ascend C API catalog"
        if path.startswith("docs/api/context/"):
            return "api_reference", "api", "Per-API reference documentation"
        return "api_docs", "api", "API documentation"
    if path.startswith("docs/guide/"):
        topic = classify_guide_topic(path)
        return "guide", topic, "Programming guide and practice documentation"
    if path.startswith("docs/figures/") or path.startswith("docs/guide/figures/"):
        return "figures", "docs", "Documentation figures"
    if len(parts) > 1:
        name = Path(path).name
        if "contributing" in name:
            return "api_contribution_guide", name, "API contribution guide"
        if name == "quick_start.md":
            return "quick_start", "docs", "Project quick start"
    return "docs_overview", "docs", "Documentation overview"


def classify_guide_topic(path: str) -> str:
    if "入门教程" in path:
        return "tutorial"
    if "兼容性迁移指南" in path or "兼容性指南" in path:
        return "migration"
    if "编译与运行" in path:
        return "build_run"
    if "调试调优" in path or "功能调试" in path:
        return "debug_tuning"
    if "编程模型" in path:
        return "programming_model"
    if "语言扩展层" in path:
        return "language_extension"
    if "类库API" in path:
        return "api_usage_guide"
    if "硬件约束" in path:
        return "hardware_constraint"
    if "硬件实现" in path or "架构规格" in path:
        return "hardware_spec"
    if "工程化算子开发" in path:
        return "operator_engineering"
    if "Host侧Tiling实现" in path or "Tiling" in path:
        return "tiling"
    if "算子入图" in path or "AI框架算子适配" in path:
        return "framework_integration"
    if "CPP标准支持" in path or "语法限制" in path:
        return "cpp_support"
    if "FAQ" in path:
        return "faq"
    if "性能优化" in path or "性能分析" in path:
        return "optimization"
    if "优秀实践" in path or "算子实践参考" in path:
        return "practice_case"
    return "guide_general"


def _classify_examples(parts: tuple[str, ...]) -> tuple[str, str, str]:
    if len(parts) == 2 and parts[1] == "README.md":
        return "examples", "overview", "Ascend C examples overview"
    family = parts[1] if len(parts) > 1 else "examples"
    topic = parts[2] if len(parts) > 2 else family
    if family == "01_simd_cpp_api":
        label = "SIMD C++ API examples"
    elif family == "02_simd_c_api":
        label = "SIMD C API examples"
    elif family == "03_simt_api":
        label = "SIMT API examples"
    else:
        label = "Ascend C examples"
    return "examples", f"{family}/{topic}", label


def _classify_tests(parts: tuple[str, ...]) -> tuple[str, str, str]:
    if len(parts) >= 3:
        return "tests", f"{parts[1]}/{parts[2]}", "Tests and usage verification"
    if len(parts) >= 2:
        return "tests", parts[1], "Tests and usage verification"
    return "tests", "tests", "Tests and usage verification"


def _classify_cann_common(path: str) -> tuple[str, str, str]:
    name = Path(path).name.lower()
    if name.startswith("bookmap_"):
        return "api_catalog", "common", "CANN documentation navigation catalog"
    return "guide", "common", "CANN common reference documentation"


def _classify_cann_ops(path: str, top: str) -> tuple[str, str, str]:
    topic = top.removeprefix("ops-") or top
    name = Path(path).name.lower()
    if name == "op_api_list.md":
        return "api_catalog", topic, f"CANN {topic} operator catalog"
    return "api_reference", topic, f"CANN {topic} operator reference"
