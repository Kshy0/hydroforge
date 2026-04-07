# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Scatter expression parser for virtual fields.

Supports expressions like:
    scatter_sum(subcell_runoff * subcell_weight, subcell_cell_idx)
    scatter_mean(subcell_runoff * subcell_weight, subcell_cell_idx)
    scatter_sum(snow.runoff * subcell_weight, subcell_cell_idx)

These denote many-to-one aggregation operations where source tensors
(dimension M) are reduced to target tensors (dimension N) via an
index mapping tensor.  Dotted qualified names (``module.field``) are
supported so that tokens can be disambiguated across modules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional, Set, Tuple

# Regex for a single identifier token that may be qualified with a dot
# e.g.  ``snow.runoff``  or plain  ``subcell_weight``.
TOKEN_RE = r'\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*\b'

# Compile the *identifier* pattern used by :func:`extract_tokens`.
_TOKEN_PROG = re.compile(TOKEN_RE)

# Regex that matches  scatter_sum(..., idx)  or  scatter_mean(..., idx)
# The index variable may now be a dotted qualified name.
_SCATTER_RE = re.compile(
    r'^scatter_(sum|mean)\(\s*(.+)\s*,\s*([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)\s*\)$',
    re.DOTALL,
)


def extract_tokens(expr: str) -> Set[str]:
    """Extract all identifier tokens (possibly dotted) from *expr*."""
    return set(_TOKEN_PROG.findall(expr))


@dataclass(frozen=True)
class ScatterExpr:
    """Parsed scatter expression."""
    mode: Literal["sum", "mean"]  # scatter_sum or scatter_mean
    value_expr: str               # elementwise expression over source tensors
    index_var: str                # mapping tensor name  (M,) -> target indices

    @property
    def value_tokens(self) -> Set[str]:
        """Extract all identifier tokens (possibly dotted) from the value expression."""
        return extract_tokens(self.value_expr)


def parse_scatter_expr(expr: str) -> Optional[ScatterExpr]:
    """
    Try to parse *expr* as a scatter expression.

    Returns ``ScatterExpr`` on success, ``None`` if *expr* is a plain
    elementwise expression (no ``scatter_`` prefix).

    Raises ``ValueError`` on malformed scatter expressions.
    """
    expr = expr.strip()
    if not expr.startswith("scatter_"):
        return None

    m = _SCATTER_RE.match(expr)
    if m is None:
        raise ValueError(
            f"Malformed scatter expression: '{expr}'. "
            "Expected: scatter_sum(<value_expr>, <index_var>) or "
            "scatter_mean(<value_expr>, <index_var>)"
        )

    mode = m.group(1)  # "sum" or "mean"
    value_expr = m.group(2).strip()
    index_var = m.group(3).strip()

    if not value_expr:
        raise ValueError("scatter expression has empty value expression")
    if not re.match(r'^[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*$', index_var):
        raise ValueError(f"scatter index must be a simple or dotted identifier, got: '{index_var}'")

    return ScatterExpr(mode=mode, value_expr=value_expr, index_var=index_var)
