"""Declarative data for compiled CUDA bindings."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CudaExtensionSpec:
    source: Path
    functions: tuple[str, ...] = ()
    declarations: tuple[str, ...] = ()
    cflags: tuple[str, ...] = ("-O3", "--use_fast_math")
    source_prefixes: tuple[Path, ...] = ()
    inline_includes: tuple[Path, ...] = ()
    cpp_headers: tuple[str, ...] = ()
    include_paths: tuple[Path, ...] = ()
    ldflags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source, Path):
            raise TypeError("CUDA extension source must be a pathlib.Path")
        tuple_fields = {
            "functions": (self.functions, str),
            "declarations": (self.declarations, str),
            "cflags": (self.cflags, str),
            "source_prefixes": (self.source_prefixes, Path),
            "inline_includes": (self.inline_includes, Path),
            "cpp_headers": (self.cpp_headers, str),
            "include_paths": (self.include_paths, Path),
            "ldflags": (self.ldflags, str),
        }
        for name, (values, element_type) in tuple_fields.items():
            if type(values) is not tuple:
                raise TypeError(f"CUDA extension {name} must be an exact tuple")
            invalid = [
                type(value).__name__
                for value in values if not isinstance(value, element_type)
            ]
            if invalid:
                raise TypeError(
                    f"CUDA extension {name} elements must be "
                    f"{element_type.__name__}: {invalid}"
                )
        if any(not name.isidentifier() for name in self.functions):
            raise ValueError(
                "CUDA extension functions must be Python/C++ identifiers"
            )
        for name in (
            "functions", "cflags", "source_prefixes", "cpp_headers",
            "include_paths", "ldflags",
        ):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"CUDA extension {name} must be unique")
        if any(not value for value in (*self.cflags, *self.cpp_headers, *self.ldflags)):
            raise ValueError("CUDA extension flags and headers must be non-empty")
        if self.declarations and len(self.declarations) != len(self.functions):
            raise ValueError(
                "CUDA extension declarations must correspond one-to-one with "
                "functions"
            )

    def materialize_source(self) -> str:
        names = [path.name for path in self.inline_includes]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                "CUDA inline include basenames must be unique: "
                f"{duplicates}"
            )
        includes = dict(zip(names, self.inline_includes, strict=True))
        emitted: set[Path] = set()

        def expand(text: str) -> str:
            for name, path in includes.items():
                token = f'#include "{name}"'
                while token in text:
                    if path in emitted:
                        replacement = ""
                    else:
                        emitted.add(path)
                        replacement = expand(path.read_text())
                    text = text.replace(token, replacement, 1)
            return text

        source = ""
        for path in (*self.source_prefixes, self.source):
            if source and not source.endswith("\n"):
                source += "\n"
            source += path.read_text()
        source = expand(source)
        unused = sorted(
            str(path) for path in self.inline_includes if path not in emitted
        )
        if unused:
            raise ValueError(f"CUDA inline includes are not referenced: {unused}")
        unresolved = sorted(set(re.findall(
            r'^\s*#include\s+"([^"]+)"', source, re.MULTILINE,
        )))
        if unresolved:
            raise ValueError(
                "CUDA quoted includes must be declared through "
                f"inline_includes: {unresolved}"
            )
        return source


def cuda_declarations(source: str, functions: Sequence[str]) -> tuple[str, ...]:
    declarations = []
    for function in functions:
        cuda_function_signature(source, function)
        match = re.search(
            rf"(?m)^void\s+{re.escape(function)}\s*\((.*?)\)\s*\{{",
            source, re.DOTALL,
        )
        if match is None:
            raise ValueError(f"CUDA source does not define {function}()")
        declarations.append(f"void {function}({match.group(1)});")
    return tuple(declarations)


def cuda_function_signature(
    source: str, function: str,
) -> tuple[tuple[str, str], ...]:
    """Return exact ``(name, normalized type)`` launcher parameters."""
    match = re.search(
        rf"(?m)^void\s+{re.escape(function)}\s*\((.*?)\)\s*\{{",
        source, re.DOTALL,
    )
    if match is None:
        raise ValueError(f"CUDA source does not define {function}()")
    declaration = match.group(1).strip()
    if not declaration:
        return ()
    parameters: list[tuple[str, str]] = []
    depth = 0
    start = 0
    chunks = []
    for index, character in enumerate(declaration):
        if character in "<([":
            depth += 1
        elif character in ">)]":
            if depth == 0:
                raise ValueError(
                    f"unbalanced parameter delimiters in {function}()"
                )
            depth -= 1
        elif character == "," and depth == 0:
            chunks.append(declaration[start:index])
            start = index + 1
    if depth != 0:
        raise ValueError(f"unbalanced parameter delimiters in {function}()")
    chunks.append(declaration[start:])
    for chunk in chunks:
        parameter = chunk.strip()
        if not parameter:
            raise ValueError(f"empty parameter in {function}()")
        if "=" in parameter:
            raise ValueError(
                f"CUDA launcher {function}() may not define default "
                "arguments; every value is owned by KernelSpec/model binding"
            )
        name = re.search(r"([A-Za-z_]\w*)\s*$", parameter)
        if name is None:
            raise ValueError(
                f"cannot parse parameter in {function}(): {chunk.strip()!r}"
            )
        parameter_name = name.group(1)
        native_type = parameter[:name.start()].strip()
        native_type = re.sub(r"\s+", " ", native_type)
        if not native_type:
            raise ValueError(
                f"cannot parse parameter type in {function}(): "
                f"{chunk.strip()!r}"
            )
        parameters.append((parameter_name, native_type))
    names = tuple(name for name, _native_type in parameters)
    if len(names) != len(set(names)):
        raise ValueError(f"CUDA function {function}() has duplicate parameters")
    return tuple(parameters)


def cuda_function_parameters(source: str, function: str) -> tuple[str, ...]:
    """Return exact C++ parameter names for one native launch definition."""

    return tuple(
        name for name, _native_type
        in cuda_function_signature(source, function)
    )


def cuda_narrowed_index_parameters(
    source: str, function: str, index_parameters: Sequence[str],
) -> tuple[str, ...]:
    """Find explicit signed-64 to signed-32 casts in one launcher body.

    Canonical ``index`` parameters have an exact int64 native ABI.  A wrapper
    that immediately casts one to ``int`` either declared the wrong semantic
    kind (it should be ``int32``) or silently truncates a real index.  Detect
    both C-style and ``static_cast`` spellings before compiling the extension.
    """

    match = re.search(
        rf"(?m)^void\s+{re.escape(function)}\s*\((.*?)\)\s*\{{",
        source, re.DOTALL,
    )
    if match is None:
        raise ValueError(f"CUDA source does not define {function}()")
    start = match.end() - 1
    depth = 0
    end = None
    for offset, character in enumerate(source[start:], start=start):
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                end = offset
                break
    if end is None:
        raise ValueError(f"CUDA function {function}() has an unclosed body")
    body = source[start + 1:end]
    narrowed = []
    for name in index_parameters:
        escaped = re.escape(name)
        if re.search(rf"\(\s*int\s*\)\s*{escaped}\b", body) or re.search(
            rf"static_cast\s*<\s*int\s*>\s*\(\s*{escaped}\s*\)", body,
        ):
            narrowed.append(name)
    return tuple(narrowed)
