"""Deterministic names shared by compilers and output schemas."""

import re
from pathlib import Path
from typing import Annotated, Any, TypeAlias

from pydantic import AfterValidator


def _validate_identifier(value: str) -> str:
    if not value.isidentifier():
        raise ValueError("must be a valid identifier")
    return value


def _validate_dotted_path(value: str) -> str:
    if any(not name.isidentifier() for name in value.split(".")):
        raise ValueError("must be a dotted attribute path")
    return value


Identifier: TypeAlias = Annotated[str, AfterValidator(_validate_identifier)]
DottedPath: TypeAlias = Annotated[str, AfterValidator(_validate_dotted_path)]


def sanitize_symbol(name: str) -> str:
    """Return a stable Python/C/filename-safe spelling."""

    for operator, spelling in (
        ("**", "_pow_"),
        ("^", "_pow_"),
        ("+", "_plus_"),
        ("-", "_minus_"),
        ("*", "_mul_"),
        ("/", "_div_"),
        (".", "_dot_"),
    ):
        name = name.replace(operator, spelling)
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if name and name[0].isdigit():
        name = "_" + name
    return name


def validate_safe_path_component(value: Any, *, label: str) -> str:
    """Return one exact filename component with no traversal semantics."""

    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty exact string")
    if (
        value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{label} must be one safe path component")
    return value


def validate_netcdf_name(value: str) -> str:
    """Validate one NetCDF variable/dimension name, including Unicode names."""

    if (
        not value
        or len(value.encode("utf-8")) > 256
        or not (
            value[0].isascii()
            and (value[0].isalnum() or value[0] == "_")
            or ord(value[0]) >= 128
        )
        or value.endswith(" ")
        or "/" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError(f"invalid NetCDF name {value!r}")
    return value
