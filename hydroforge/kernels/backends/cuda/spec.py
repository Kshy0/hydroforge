"""Declarative device-only CUDA sources."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Annotated

from pydantic import Field, ValidationInfo, field_validator

from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.kernels.backends.cuda.rtc import RtcProgram
from hydroforge.kernels.math_mode import fast_math

_NonemptyText = Annotated[str, Field(min_length=1)]


def cuda_physics_options(options: tuple[str, ...]) -> tuple[str, ...]:
    """Runtime-compiler options of a physics program under the math mode."""

    if fast_math() and "--use_fast_math" not in options:
        return ("--use_fast_math", *options)
    return options


class CudaExtensionSpec(HydroForgeModel):
    """One device-only source compiled at runtime.

    ``options`` use the runtime compiler's spelling (for example
    ``--ftz=false``, ``-DNAME=value``); optimization is on by default. The math
    mode (:mod:`hydroforge.kernels.math_mode`) decides fast math, so sources
    do not set it.
    """

    source: Path = Field(strict=False)
    options: tuple[_NonemptyText, ...] = ()
    source_prefixes: tuple[Annotated[Path, Field(strict=False)], ...] = ()
    inline_includes: tuple[Annotated[Path, Field(strict=False)], ...] = ()
    include_root: Path | None = Field(default=None, strict=False)

    @field_validator("options", "source_prefixes")
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

    def program(self, name: str) -> RtcProgram:
        return RtcProgram(
            self._materialize_source(), cuda_physics_options(self.options), name
        )

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
