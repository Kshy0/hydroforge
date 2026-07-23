"""Deterministic names shared by compilers and output schemas."""

import re


def sanitize_symbol(name: str) -> str:
    """Return a stable Python/C/filename-safe spelling."""

    for operator, spelling in (
        ("**", "_pow_"), ("^", "_pow_"), ("+", "_plus_"),
        ("-", "_minus_"), ("*", "_mul_"), ("/", "_div_"),
        (".", "_dot_"),
    ):
        name = name.replace(operator, spelling)
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if name and name[0].isdigit():
        name = "_" + name
    return re.sub(r"_+", "_", name).strip("_")
