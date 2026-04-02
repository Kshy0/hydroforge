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

These denote many-to-one aggregation operations where source tensors
(dimension M) are reduced to target tensors (dimension N) via an
index mapping tensor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional, Set, Tuple

# Regex that matches  scatter_sum(..., idx)  or  scatter_mean(..., idx)
# Captures:  group(1)=mode  group(2)=value_expr  group(3)=index_tensor
_SCATTER_RE = re.compile(
    r'^scatter_(sum|mean)\(\s*(.+)\s*,\s*([a-zA-Z_]\w*)\s*\)$',
    re.DOTALL,
)


@dataclass(frozen=True)
class ScatterExpr:
    """Parsed scatter expression."""
    mode: Literal["sum", "mean"]  # scatter_sum or scatter_mean
    value_expr: str               # elementwise expression over source tensors
    index_var: str                # mapping tensor name  (M,) -> target indices

    @property
    def value_tokens(self) -> Set[str]:
        """Extract all identifier tokens from the value expression."""
        return set(re.findall(r'\b[a-zA-Z_]\w*\b', self.value_expr))


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
    if not re.match(r'^[a-zA-Z_]\w*$', index_var):
        raise ValueError(f"scatter index must be a simple identifier, got: '{index_var}'")

    return ScatterExpr(mode=mode, value_expr=value_expr, index_var=index_var)
