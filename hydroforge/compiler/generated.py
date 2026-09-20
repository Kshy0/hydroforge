"""Inspectable lifetime for trusted, internally generated Python source."""

from __future__ import annotations

import linecache
import sys
from types import ModuleType


def compile_generated_module(source: str, *, name: str) -> ModuleType:
    """Register a uniquely named module and roll back failed execution."""
    filename = f"<{name}>"
    lines = source.splitlines(keepends=True)
    if source and not source.endswith("\n"):
        lines[-1] += "\n"
    linecache.cache[filename] = (len(source), None, lines, filename)
    module = ModuleType(name)
    module.__file__ = filename
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(source, filename, "exec"), module.__dict__)
    except BaseException:
        release_generated_module(name, filename)
        raise
    return module


def release_generated_module(name: str, filename: str) -> None:
    """Drop one owner's generated module and its inspectable source."""
    sys.modules.pop(name, None)
    linecache.cache.pop(filename, None)
