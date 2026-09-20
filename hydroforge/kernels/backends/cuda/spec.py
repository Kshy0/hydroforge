"""Declarative data for compiled CUDA bindings."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, ValidationInfo, field_validator

from hydroforge.contracts.validation import HydroForgeModel

_NonemptyText = Annotated[str, Field(min_length=1)]


class CudaExtensionSpec(HydroForgeModel):
    source: Path
    cflags: tuple[_NonemptyText, ...] = ("-O3", "--use_fast_math")
    source_prefixes: tuple[Path, ...] = ()
    inline_includes: tuple[Path, ...] = ()
    include_root: Path | None = None
    cpp_headers: tuple[_NonemptyText, ...] = ()
    include_paths: tuple[Path, ...] = ()
    ldflags: tuple[_NonemptyText, ...] = ()

    @field_validator(
        "cflags", "source_prefixes", "cpp_headers", "include_paths", "ldflags"
    )
    @classmethod
    def _unique_entries(cls, values: tuple, info: ValidationInfo) -> tuple:
        if len(values) != len(set(values)):
            raise ValueError(f"CUDA extension {info.field_name} must be unique")
        return values

    @field_validator("inline_includes")
    @classmethod
    def _unique_include_names(cls, paths: tuple[Path, ...]) -> tuple[Path, ...]:
        counts = Counter(path.name for path in paths)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(
                f"CUDA inline include basenames must be unique: {duplicates}"
            )
        return paths

    def _materialize_source(self) -> str:
        includes = {path.name: path for path in self.inline_includes}
        emitted: set[Path] = set()
        root = None if self.include_root is None else self.include_root.resolve()
        if root is not None and not root.is_dir():
            raise ValueError(f"CUDA include_root is not a directory: {root}")
        include_pattern = re.compile(
            r'^\s*#include\s+"([^"]+)"',
            re.MULTILINE,
        )

        def expand(text: str, origin: Path) -> str:
            def replace(match: re.Match[str]) -> str:
                name = match.group(1)
                path = includes.get(name)
                if path is None and root is not None:
                    path = (origin.parent / name).resolve()
                    if not path.is_relative_to(root):
                        raise ValueError(
                            f"CUDA include {name!r} escapes include_root {root}"
                        )
                if path is None:
                    return match.group(0)
                path = path.resolve()
                if path in emitted:
                    return ""
                emitted.add(path)
                return expand(path.read_text(), path)

            return include_pattern.sub(replace, text)

        source = ""
        for path in (*self.source_prefixes, self.source):
            if source and not source.endswith("\n"):
                source += "\n"
            source += expand(path.read_text(), path)
        unused = sorted(
            str(path) for path in self.inline_includes if path.resolve() not in emitted
        )
        if unused:
            raise ValueError(f"CUDA inline includes are not referenced: {unused}")
        unresolved = sorted(set(include_pattern.findall(source)))
        if unresolved:
            raise ValueError(
                "CUDA quoted includes must be declared through "
                f"inline_includes: {unresolved}"
            )
        return source


@dataclass(frozen=True, slots=True)
class _CompiledCudaExtension:
    """Immutable source/build plan produced by one validated group."""

    source: str
    functions: tuple[str, ...]
    declarations: tuple[str, ...]
    cflags: tuple[str, ...]
    cpp_headers: tuple[str, ...]
    include_paths: tuple[Path, ...]
    ldflags: tuple[str, ...]

    def loader_arguments(self, name: str, env_prefix: str) -> dict[str, Any]:
        """Build isolated loader options for this immutable source/ABI plan."""
        return {
            "name": name,
            "cpp_sources": "\n".join((*self.cpp_headers, *self.declarations)),
            "cuda_sources": self.source,
            "functions": self.functions,
            "extra_cuda_cflags": self.cflags,
            "extra_include_paths": tuple(map(str, self.include_paths)),
            "extra_ldflags": self.ldflags,
            "env_prefix": env_prefix,
        }


def _function_definition(source: str, function: str) -> re.Match[str]:
    match = re.search(
        rf"(?m)^void\s+{re.escape(function)}\s*\((.*?)\)\s*\{{",
        source,
        re.DOTALL,
    )
    if match is None:
        raise ValueError(f"CUDA source does not define {function}()")
    return match


def cuda_declarations(source: str, functions: Sequence[str]) -> tuple[str, ...]:
    declarations = []
    for function in functions:
        definition = _function_definition(source, function).group(1)
        _parse_parameters(definition, function)
        declarations.append(f"void {function}({definition});")
    return tuple(declarations)


def cuda_function_signature(
    source: str,
    function: str,
) -> tuple[tuple[str, str], ...]:
    """Return exact ``(name, normalized type)`` launcher parameters."""
    return _parse_parameters(_function_definition(source, function).group(1), function)


def _parse_parameters(declaration: str, function: str) -> tuple[tuple[str, str], ...]:
    declaration = declaration.strip()
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
                raise ValueError(f"unbalanced parameter delimiters in {function}()")
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
        native_type = parameter[: name.start()].strip()
        native_type = re.sub(r"\s+", " ", native_type)
        if not native_type:
            raise ValueError(
                f"cannot parse parameter type in {function}(): {chunk.strip()!r}"
            )
        parameters.append((parameter_name, native_type))
    names = tuple(name for name, _native_type in parameters)
    if len(names) != len(set(names)):
        raise ValueError(f"CUDA function {function}() has duplicate parameters")
    return tuple(parameters)


def cuda_narrowed_index_parameters(
    source: str,
    function: str,
    index_parameters: Sequence[str],
) -> tuple[str, ...]:
    """Find signed-64 to signed-32 conversions in one launcher body.

    Canonical ``index`` parameters have an exact int64 native ABI.  A wrapper
    that converts one to a signed 32-bit value either declared the wrong
    semantic kind (it should be ``int32``) or silently truncates a real index.
    Reject casts, functional construction and local-variable initialization
    before compiling the extension.
    """

    def code_only(text: str) -> str:
        pattern = re.compile(
            r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|"
            r"'(?:\\.|[^'\\])*'",
            re.DOTALL,
        )

        def mask(match: re.Match[str]) -> str:
            return "".join(
                "\n" if character == "\n" else " " for character in match.group(0)
            )

        return pattern.sub(mask, text)

    code = code_only(source)
    match = _function_definition(code, function)
    start = match.end() - 1
    depth = 0
    end = None
    for offset, character in enumerate(code[start:], start=start):
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                end = offset
                break
    if end is None:
        raise ValueError(f"CUDA function {function}() has an unclosed body")
    body = code[start + 1 : end]
    signed_int32 = r"(?:signed\s+int|int|(?:std\s*::\s*)?int32_t)"

    def split_arguments(arguments: str) -> tuple[str, ...]:
        if not arguments.strip():
            return ()
        chunks: list[str] = []
        start = 0
        pairs = {")": "(", "]": "[", "}": "{"}
        stack: list[str] = []
        for offset, character in enumerate(arguments):
            if character in "([{":
                stack.append(character)
            elif character in ")]}":
                if not stack or stack[-1] != pairs[character]:
                    raise ValueError(
                        f"unbalanced call arguments in CUDA launcher {function}()"
                    )
                stack.pop()
            elif character == "," and not stack:
                chunks.append(arguments[start:offset].strip())
                start = offset + 1
        if stack:
            raise ValueError(f"unbalanced call arguments in CUDA launcher {function}()")
        chunks.append(arguments[start:].strip())
        return tuple(chunks)

    def calls(name: str) -> tuple[tuple[str, ...], ...]:
        found: list[tuple[str, ...]] = []
        for call in re.finditer(rf"\b{re.escape(name)}\s*\(", body):
            open_paren = body.find("(", call.start())
            depth = 0
            close_paren = None
            for offset in range(open_paren, len(body)):
                character = body[offset]
                if character == "(":
                    depth += 1
                elif character == ")":
                    depth -= 1
                    if depth == 0:
                        close_paren = offset
                        break
            if close_paren is None:
                raise ValueError(
                    f"call to {name}() in CUDA launcher {function}() "
                    "has unbalanced parentheses"
                )
            found.append(split_arguments(body[open_paren + 1 : close_paren]))
        return tuple(found)

    helper_signatures: dict[str, tuple[tuple[str, str], ...]] = {}
    for helper in re.finditer(
        r"(?m)^void\s+([A-Za-z_]\w*)\s*\((.*?)\)\s*\{",
        code,
        re.DOTALL,
    ):
        helper_name = helper.group(1)
        if helper_name == function or helper_name in helper_signatures:
            continue
        helper_signatures[helper_name] = _parse_parameters(helper.group(2), helper_name)

    local_int32_names = tuple(re.findall(rf"\b{signed_int32}\s+([A-Za-z_]\w*)\b", body))
    int32_arguments: list[str] = []
    for helper_name, signature in helper_signatures.items():
        positions = [
            index
            for index, (_name, native_type) in enumerate(signature)
            if re.fullmatch(
                signed_int32,
                re.sub(r"\b(?:const|volatile)\b", "", native_type)
                .strip()
                .rstrip("&")
                .strip(),
            )
        ]
        for arguments in calls(helper_name):
            if len(arguments) == len(signature):
                int32_arguments.extend(arguments[index] for index in positions)

    narrowed = []
    for name in index_parameters:
        escaped = re.escape(name)
        local_assignment = any(
            re.search(
                rf"\b{re.escape(local)}\s*(?:=(?!=)|[+\-*/%]=)"
                rf"\s*[^;]*\b{escaped}\b",
                body,
            )
            for local in local_int32_names
        )
        direct_conversion = any(
            re.search(pattern, body)
            for pattern in (
                rf"\(\s*{signed_int32}\s*\)\s*{escaped}\b",
                rf"static_cast\s*<\s*{signed_int32}\s*>\s*"
                rf"\(\s*{escaped}\b",
                rf"\b{signed_int32}\s*[({{]\s*{escaped}\b",
            )
        )
        call_conversion = any(
            re.search(rf"\b{escaped}\b", argument) for argument in int32_arguments
        )
        if direct_conversion or local_assignment or call_conversion:
            narrowed.append(name)
    return tuple(narrowed)
