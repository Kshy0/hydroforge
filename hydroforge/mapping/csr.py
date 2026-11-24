# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Zero-SciPy CSR sparse matrix and exact KD-tree nearest-neighbor search.

Implemented with NumPy and Numba ``@njit`` so :mod:`hydroforge.mapping` has no
``scipy`` dependency while keeping exact float64 coalescing, canonical CSR
invariants, immutable storage freezing, parallel SpMM, and exact Euclidean /
unit-sphere nearest-neighbor queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit, prange

from hydroforge.core.arrays import immutable_array

# --------------------------------------------------------------------------- #
# Numba CSR kernels
# --------------------------------------------------------------------------- #


@njit(cache=True)
def _csr_from_coo_kernel(
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nnz = rows.size
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    for k in range(nnz):
        indptr[rows[k] + 1] += 1
    for r in range(n_rows):
        indptr[r + 1] += indptr[r]

    out_cols = np.empty(nnz, dtype=np.int64)
    out_data = np.empty(nnz, dtype=data.dtype)
    cursor = indptr[:-1].copy()
    for k in range(nnz):
        r = rows[k]
        pos = cursor[r]
        out_cols[pos] = cols[k]
        out_data[pos] = data[k]
        cursor[r] = pos + 1
    return indptr, out_cols, out_data


@njit(cache=True)
def _heapsort_slice(
    indices: np.ndarray, data: np.ndarray, start: int, end: int
) -> None:
    length = end - start
    if length <= 1:
        return
    if length <= 32:
        for i in range(start + 1, end):
            key = indices[i]
            val = data[i]
            j = i - 1
            while j >= start and indices[j] > key:
                indices[j + 1] = indices[j]
                data[j + 1] = data[j]
                j -= 1
            indices[j + 1] = key
            data[j + 1] = val
        return

    for root in range(length // 2 - 1, -1, -1):
        i = root
        while True:
            child = 2 * i + 1
            if child >= length:
                break
            if (
                child + 1 < length
                and indices[start + child] < indices[start + child + 1]
            ):
                child += 1
            if indices[start + i] < indices[start + child]:
                ti = indices[start + i]
                indices[start + i] = indices[start + child]
                indices[start + child] = ti
                td = data[start + i]
                data[start + i] = data[start + child]
                data[start + child] = td
                i = child
            else:
                break
    for last in range(length - 1, 0, -1):
        ti = indices[start]
        indices[start] = indices[start + last]
        indices[start + last] = ti
        td = data[start]
        data[start] = data[start + last]
        data[start + last] = td
        i = 0
        while True:
            child = 2 * i + 1
            if child >= last:
                break
            if (
                child + 1 < last
                and indices[start + child] < indices[start + child + 1]
            ):
                child += 1
            if indices[start + i] < indices[start + child]:
                ti = indices[start + i]
                indices[start + i] = indices[start + child]
                indices[start + child] = ti
                td = data[start + i]
                data[start + i] = data[start + child]
                data[start + child] = td
                i = child
            else:
                break


@njit(cache=True)
def _csr_sort_indices_kernel(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray
) -> None:
    n_rows = indptr.size - 1
    for r in range(n_rows):
        a = indptr[r]
        b = indptr[r + 1]
        if b - a > 1:
            _heapsort_slice(indices, data, a, b)


@njit(cache=True)
def _csr_check_format_kernel(
    indptr: np.ndarray, indices: np.ndarray
) -> tuple[bool, bool]:
    n_rows = indptr.size - 1
    sorted_ok = True
    no_dups = True
    for r in range(n_rows):
        a = indptr[r]
        b = indptr[r + 1]
        for k in range(a + 1, b):
            prev = indices[k - 1]
            cur = indices[k]
            if cur < prev:
                sorted_ok = False
                no_dups = False
                return sorted_ok, no_dups
            if cur == prev:
                no_dups = False
    return sorted_ok, no_dups


@njit(cache=True)
def _csr_sum_duplicates_kernel(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = indptr.size - 1
    new_indptr = np.empty(n_rows + 1, dtype=np.int64)
    new_indptr[0] = 0
    write = 0
    for r in range(n_rows):
        a = indptr[r]
        b = indptr[r + 1]
        if b > a:
            _heapsort_slice(indices, data, a, b)
            cur_col = indices[a]
            cur_val = data[a]
            for k in range(a + 1, b):
                col = indices[k]
                if col == cur_col:
                    cur_val += data[k]
                else:
                    indices[write] = cur_col
                    data[write] = cur_val
                    write += 1
                    cur_col = col
                    cur_val = data[k]
            indices[write] = cur_col
            data[write] = cur_val
            write += 1
        new_indptr[r + 1] = write
    return new_indptr, indices[:write].copy(), data[:write].copy()


@njit(cache=True)
def _csr_eliminate_zeros_kernel(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = indptr.size - 1
    new_indptr = np.empty(n_rows + 1, dtype=np.int64)
    new_indptr[0] = 0
    out_indices = np.empty(data.size, dtype=np.int64)
    out_data = np.empty(data.size, dtype=data.dtype)
    write = 0
    for r in range(n_rows):
        a = indptr[r]
        b = indptr[r + 1]
        for k in range(a, b):
            v = data[k]
            if v != 0.0:
                out_indices[write] = indices[k]
                out_data[write] = v
                write += 1
        new_indptr[r + 1] = write
    return new_indptr, out_indices[:write].copy(), out_data[:write].copy()


@njit(cache=True)
def _csr_transpose_kernel(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, n_cols: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = indptr.size - 1
    nnz = data.size
    t_indptr = np.zeros(n_cols + 1, dtype=np.int64)
    for k in range(nnz):
        t_indptr[indices[k] + 1] += 1
    for c in range(n_cols):
        t_indptr[c + 1] += t_indptr[c]

    t_indices = np.empty(nnz, dtype=np.int64)
    t_data = np.empty(nnz, dtype=data.dtype)
    cursor = t_indptr[:-1].copy()
    for r in range(n_rows):
        for k in range(indptr[r], indptr[r + 1]):
            c = indices[k]
            pos = cursor[c]
            t_indices[pos] = r
            t_data[pos] = data[k]
            cursor[c] = pos + 1
    return t_indptr, t_indices, t_data


@njit(cache=True)
def _csr_row_slice_kernel(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_sel = rows.size
    out_indptr = np.empty(n_sel + 1, dtype=np.int64)
    out_indptr[0] = 0
    total = 0
    for i in range(n_sel):
        r = rows[i]
        total += indptr[r + 1] - indptr[r]
        out_indptr[i + 1] = total
    out_indices = np.empty(total, dtype=np.int64)
    out_data = np.empty(total, dtype=data.dtype)
    pos = 0
    for i in range(n_sel):
        r = rows[i]
        a = indptr[r]
        b = indptr[r + 1]
        for k in range(a, b):
            out_indices[pos] = indices[k]
            out_data[pos] = data[k]
            pos += 1
    return out_indptr, out_indices, out_data


@njit(cache=True)
def _csr_row_sums_kernel(indptr: np.ndarray, data: np.ndarray) -> np.ndarray:
    n_rows = indptr.size - 1
    out = np.zeros(n_rows, dtype=np.float64)
    for r in range(n_rows):
        acc = 0.0
        for k in range(indptr[r], indptr[r + 1]):
            acc += data[k]
        out[r] = acc
    return out


@njit(cache=True)
def _csr_col_sums_kernel(
    indices: np.ndarray, data: np.ndarray, n_cols: int
) -> np.ndarray:
    out = np.zeros(n_cols, dtype=np.float64)
    for k in range(data.size):
        out[indices[k]] += data[k]
    return out


@njit(cache=True)
def _csr_scale_rows_kernel(
    indptr: np.ndarray, data: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    out = np.empty(data.size, dtype=np.float64)
    n_rows = indptr.size - 1
    for r in range(n_rows):
        s = scale[r]
        for k in range(indptr[r], indptr[r + 1]):
            out[k] = data[k] * s
    return out


@njit(cache=True)
def _csr_add_sorted_kernel(
    a_ptr: np.ndarray,
    a_idx: np.ndarray,
    a_dat: np.ndarray,
    b_ptr: np.ndarray,
    b_idx: np.ndarray,
    b_dat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = a_ptr.size - 1
    max_nnz = a_dat.size + b_dat.size
    out_ptr = np.empty(n_rows + 1, dtype=np.int64)
    out_ptr[0] = 0
    out_idx = np.empty(max_nnz, dtype=np.int64)
    out_dat = np.empty(max_nnz, dtype=np.float64)
    pos = 0
    for r in range(n_rows):
        ia = a_ptr[r]
        ea = a_ptr[r + 1]
        ib = b_ptr[r]
        eb = b_ptr[r + 1]
        while ia < ea and ib < eb:
            ca = a_idx[ia]
            cb = b_idx[ib]
            if ca < cb:
                out_idx[pos] = ca
                out_dat[pos] = a_dat[ia]
                ia += 1
            elif cb < ca:
                out_idx[pos] = cb
                out_dat[pos] = b_dat[ib]
                ib += 1
            else:
                out_idx[pos] = ca
                out_dat[pos] = a_dat[ia] + b_dat[ib]
                ia += 1
                ib += 1
            pos += 1
        while ia < ea:
            out_idx[pos] = a_idx[ia]
            out_dat[pos] = a_dat[ia]
            ia += 1
            pos += 1
        while ib < eb:
            out_idx[pos] = b_idx[ib]
            out_dat[pos] = b_dat[ib]
            ib += 1
            pos += 1
        out_ptr[r + 1] = pos
    return out_ptr, out_idx[:pos].copy(), out_dat[:pos].copy()


@njit(cache=True, parallel=True)
def _csr_spmm_2d_kernel(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, x: np.ndarray
) -> np.ndarray:
    n_rows = indptr.size - 1
    n_cols_out = x.shape[1]
    out = np.zeros((n_rows, n_cols_out), dtype=np.float64)
    for r in prange(n_rows):
        a = indptr[r]
        b = indptr[r + 1]
        for k in range(a, b):
            w = float(data[k])
            col = indices[k]
            for j in range(n_cols_out):
                out[r, j] += w * x[col, j]
    return out


@njit(cache=True, parallel=True)
def _csr_spmv_1d_kernel(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, x: np.ndarray
) -> np.ndarray:
    n_rows = indptr.size - 1
    out = np.zeros(n_rows, dtype=np.float64)
    for r in prange(n_rows):
        acc = 0.0
        for k in range(indptr[r], indptr[r + 1]):
            acc += float(data[k]) * x[indices[k]]
        out[r] = acc
    return out


@dataclass(frozen=True, slots=True)
class _COOView:
    row: np.ndarray
    col: np.ndarray
    data: np.ndarray
    shape: tuple[int, int]


class CSRMatrix:
    """Lightweight, canonical-aware CSR sparse matrix with optional immutability."""

    _IMMUTABLE_ATTRIBUTES = frozenset({"data", "indices", "indptr", "_shape"})

    def __init__(
        self,
        arg1: Any,
        *,
        shape: tuple[int, int] | None = None,
        dtype: Any = None,
        copy: bool = False,
    ) -> None:
        self._hydroforge_sealed = False
        if isinstance(arg1, CSRMatrix):
            resolved_dtype = np.dtype(dtype) if dtype is not None else arg1.dtype
            copy_flag = True if copy else None
            self.data = np.array(arg1.data, dtype=resolved_dtype, copy=copy_flag)
            self.indices = np.array(arg1.indices, dtype=np.int64, copy=copy_flag)
            self.indptr = np.array(arg1.indptr, dtype=np.int64, copy=copy_flag)
            self._shape = (
                arg1.shape if shape is None else (int(shape[0]), int(shape[1]))
            )
            return

        if isinstance(arg1, tuple) and len(arg1) == 3:
            raw_data, raw_indices, raw_indptr = arg1
            data_arr = (
                np.asarray(raw_data, dtype=dtype)
                if dtype is not None
                else np.asarray(raw_data)
            )
            if copy:
                data_arr = data_arr.copy()
            indices_raw = np.asarray(raw_indices)
            indptr_raw = np.asarray(raw_indptr)
            for name, raw, comp in (
                ("sparse_indices", raw_indices, indices_raw),
                ("sparse_indptr", raw_indptr, indptr_raw),
            ):
                if (isinstance(raw, np.ndarray) or comp.size > 0) and comp.dtype not in {
                    np.dtype(np.int32),
                    np.dtype(np.int64),
                }:
                    raise ValueError(f"{name} must use signed int32 or int64 storage")
            copy_flag = True if copy else None
            indices_arr = np.array(indices_raw, dtype=np.int64, copy=copy_flag)
            indptr_arr = np.array(indptr_raw, dtype=np.int64, copy=copy_flag)
            if shape is None:
                n_rows = max(indptr_arr.size - 1, 0)
                n_cols = int(indices_arr.max()) + 1 if indices_arr.size else 0
                shape = (n_rows, n_cols)
            n_rows, n_cols = int(shape[0]), int(shape[1])
            if n_rows < 0 or n_cols < 0:
                raise ValueError("CSRMatrix shape must be nonnegative")
            self.data = np.ascontiguousarray(data_arr)
            self.indices = np.ascontiguousarray(indices_arr)
            self.indptr = np.ascontiguousarray(indptr_arr)
            self._shape = (n_rows, n_cols)
            return

        if (
            isinstance(arg1, tuple)
            and len(arg1) == 2
            and isinstance(arg1[1], tuple)
            and len(arg1[1]) == 2
        ):
            raw_data, (raw_rows, raw_cols) = arg1
            data_arr = np.ascontiguousarray(
                np.asarray(raw_data, dtype=dtype)
                if dtype is not None
                else np.asarray(raw_data)
            )
            rows_arr = np.ascontiguousarray(np.asarray(raw_rows, dtype=np.int64))
            cols_arr = np.ascontiguousarray(np.asarray(raw_cols, dtype=np.int64))
            if not (data_arr.size == rows_arr.size == cols_arr.size):
                raise ValueError("COO data, row, and col lengths must match")
            if shape is None:
                n_rows = int(rows_arr.max()) + 1 if rows_arr.size else 0
                n_cols = int(cols_arr.max()) + 1 if cols_arr.size else 0
                shape = (n_rows, n_cols)
            n_rows, n_cols = int(shape[0]), int(shape[1])
            if n_rows < 0 or n_cols < 0:
                raise ValueError("CSRMatrix shape must be nonnegative")
            if rows_arr.size:
                if int(rows_arr.min()) < 0 or int(rows_arr.max()) >= n_rows:
                    raise ValueError("COO row indices fall outside matrix shape")
                if int(cols_arr.min()) < 0 or int(cols_arr.max()) >= n_cols:
                    raise ValueError("COO column indices fall outside matrix shape")
            indptr_arr, indices_arr, sorted_data = _csr_from_coo_kernel(
                rows_arr, cols_arr, data_arr, n_rows
            )
            indptr_arr, indices_arr, sorted_data = _csr_sum_duplicates_kernel(
                indptr_arr, indices_arr, sorted_data
            )
            self.data = sorted_data
            self.indices = indices_arr
            self.indptr = indptr_arr
            self._shape = (n_rows, n_cols)
            return

        raise TypeError(f"unsupported CSRMatrix initializer: {type(arg1).__name__}")

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            getattr(self, "_hydroforge_sealed", False)
            and name in self._IMMUTABLE_ATTRIBUTES
        ):
            raise TypeError("frozen mapping CSR storage is immutable")
        object.__setattr__(self, name, value)

    def _reject_mutation(self) -> None:
        if getattr(self, "_hydroforge_sealed", False):
            raise TypeError("frozen mapping CSR storage is immutable")

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def ndim(self) -> int:
        return 2

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def nnz(self) -> int:
        return int(self.data.size)

    @property
    def has_sorted_indices(self) -> bool:
        sorted_ok, _ = _csr_check_format_kernel(self.indptr, self.indices)
        return bool(sorted_ok)

    @property
    def has_canonical_format(self) -> bool:
        sorted_ok, no_dups = _csr_check_format_kernel(self.indptr, self.indices)
        return bool(sorted_ok and no_dups)

    @property
    def T(self) -> CSRMatrix:
        t_ptr, t_idx, t_dat = _csr_transpose_kernel(
            self.indptr, self.indices, self.data, self._shape[1]
        )
        return CSRMatrix(
            (t_dat, t_idx, t_ptr),
            shape=(self._shape[1], self._shape[0]),
        )

    def tocsr(self) -> CSRMatrix:
        return self

    def tocoo(self) -> _COOView:
        row = np.repeat(
            np.arange(self._shape[0], dtype=np.int64), np.diff(self.indptr)
        )
        return _COOView(
            row=row,
            col=self.indices.copy(),
            data=self.data.copy(),
            shape=self._shape,
        )

    def copy(self) -> CSRMatrix:
        return CSRMatrix(self, copy=True)

    def astype(self, dtype: Any, copy: bool = True) -> CSRMatrix:
        return CSRMatrix(self, dtype=dtype, copy=copy)

    def freeze(self) -> CSRMatrix:
        """Return an immutable view backed by owned read-only arrays."""
        frozen = CSRMatrix(
            (
                immutable_array(self.data, order="C"),
                immutable_array(self.indices, dtype=np.int64, order="C"),
                immutable_array(self.indptr, dtype=np.int64, order="C"),
            ),
            shape=self._shape,
        )
        frozen._hydroforge_sealed = True
        return frozen

    def __repr__(self) -> str:
        return (
            f"<CSRMatrix {self._shape[0]}x{self._shape[1]} "
            f"nnz={self.nnz} dtype={self.dtype}>"
        )

    def toarray(self) -> np.ndarray:
        out = np.zeros(self._shape, dtype=self.dtype)
        for r in range(self._shape[0]):
            a = int(self.indptr[r])
            b = int(self.indptr[r + 1])
            if b > a:
                np.add.at(out[r], self.indices[a:b], self.data[a:b])
        return out

    def sort_indices(self) -> None:
        if self.has_sorted_indices:
            return
        self._reject_mutation()
        if not self.indices.flags.writeable:
            self.indices = self.indices.copy()
        if not self.data.flags.writeable:
            self.data = self.data.copy()
        _csr_sort_indices_kernel(self.indptr, self.indices, self.data)

    def sum_duplicates(self) -> None:
        if self.has_canonical_format:
            return
        self._reject_mutation()
        indptr, indices, data = _csr_sum_duplicates_kernel(
            self.indptr, self.indices.copy(), self.data.copy()
        )
        self.indptr = indptr
        self.indices = indices
        self.data = data

    def eliminate_zeros(self) -> None:
        self._reject_mutation()
        indptr, indices, data = _csr_eliminate_zeros_kernel(
            self.indptr, self.indices, self.data
        )
        self.indptr = indptr
        self.indices = indices
        self.data = data

    def prune(self) -> None:
        self._reject_mutation()

    def resize(self, *shape: Any) -> None:
        self._reject_mutation()
        raise NotImplementedError("CSRMatrix.resize is not supported")

    def setdiag(self, values: Any, k: int = 0) -> None:
        del values, k
        self._reject_mutation()
        raise NotImplementedError("CSRMatrix.setdiag is not supported")

    def __setitem__(self, key: Any, value: Any) -> None:
        del key, value
        self._reject_mutation()
        raise TypeError("CSRMatrix does not support item assignment")

    def __getitem__(self, key: Any) -> CSRMatrix:
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError("CSRMatrix indexing requires a (rows, :) pair")
        row_key, col_key = key
        if col_key != slice(None):
            raise NotImplementedError("CSRMatrix only supports full-row slicing")
        if isinstance(row_key, slice):
            rows = np.arange(*row_key.indices(self._shape[0]), dtype=np.int64)
        else:
            rows = np.ascontiguousarray(np.asarray(row_key, dtype=np.int64))
            if rows.ndim != 1:
                raise ValueError("row selection must be one-dimensional")
            if rows.size:
                rows = np.where(rows < 0, rows + self._shape[0], rows)
                if np.any((rows < 0) | (rows >= self._shape[0])):
                    raise IndexError("row index out of bounds")
        out_ptr, out_idx, out_dat = _csr_row_slice_kernel(
            self.indptr, self.indices, self.data, rows
        )
        return CSRMatrix(
            (out_dat, out_idx, out_ptr),
            shape=(rows.size, self._shape[1]),
        )

    def sum(self, axis: int | None = None) -> np.ndarray | float:
        if axis is None:
            return float(np.sum(self.data, dtype=np.float64))
        if axis == 1 or axis == -1:
            return _csr_row_sums_kernel(self.indptr, self.data).reshape(
                self._shape[0], 1
            )
        if axis == 0 or axis == -2:
            return _csr_col_sums_kernel(
                self.indices, self.data, self._shape[1]
            ).reshape(1, self._shape[1])
        raise ValueError(f"invalid axis {axis} for 2D CSRMatrix")

    def multiply(self, other: Any) -> CSRMatrix:
        arr = np.asarray(other, dtype=np.float64)
        if arr.ndim == 0:
            return CSRMatrix(
                (
                    self.data.astype(np.float64) * float(arr),
                    self.indices.copy(),
                    self.indptr.copy(),
                ),
                shape=self._shape,
            )
        if arr.ndim == 2 and arr.shape == (self._shape[0], 1):
            scaled = _csr_scale_rows_kernel(
                self.indptr,
                self.data.astype(np.float64, copy=False),
                np.ascontiguousarray(arr[:, 0]),
            )
            return CSRMatrix(
                (scaled, self.indices.copy(), self.indptr.copy()),
                shape=self._shape,
            )
        raise ValueError(f"unsupported multiply shape {arr.shape} for {self._shape}")

    def __add__(self, other: CSRMatrix) -> CSRMatrix:
        if not isinstance(other, CSRMatrix):
            return NotImplemented
        if self._shape != other._shape:
            raise ValueError(f"shape mismatch: {self._shape} + {other._shape}")
        a = self if self.has_canonical_format else self.copy()
        if not a.has_canonical_format:
            a.sum_duplicates()
        b = other if other.has_canonical_format else other.copy()
        if not b.has_canonical_format:
            b.sum_duplicates()
        out_ptr, out_idx, out_dat = _csr_add_sorted_kernel(
            a.indptr,
            a.indices,
            a.data.astype(np.float64, copy=False),
            b.indptr,
            b.indices,
            b.data.astype(np.float64, copy=False),
        )
        return CSRMatrix((out_dat, out_idx, out_ptr), shape=self._shape)

    def __matmul__(self, other: np.ndarray) -> np.ndarray:
        arr = np.asarray(other, dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape[0] != self._shape[1]:
                raise ValueError(f"matmul shape mismatch: {self._shape} @ {arr.shape}")
            return _csr_spmv_1d_kernel(
                self.indptr, self.indices, self.data, np.ascontiguousarray(arr)
            )
        if arr.ndim == 2:
            if arr.shape[0] != self._shape[1]:
                raise ValueError(f"matmul shape mismatch: {self._shape} @ {arr.shape}")
            return _csr_spmm_2d_kernel(
                self.indptr, self.indices, self.data, np.ascontiguousarray(arr)
            )
        raise ValueError(f"unsupported matmul rhs ndim {arr.ndim}")


# --------------------------------------------------------------------------- #
# Exact balanced median-split KD-tree (2D planar & 3D unit-sphere geographic)
# --------------------------------------------------------------------------- #

_LEAF_SIZE = 16


@njit(cache=True)
def _quickselect_axis(
    pts: np.ndarray, order: np.ndarray, left: int, right: int, kth: int, axis: int
) -> None:
    """In-place Hoare quickselect on ``order[left:right+1]`` by ``pts[:, axis]``."""
    lo = left
    hi = right
    while lo < hi:
        mid = (lo + hi) >> 1
        # Median-of-three pivot selection avoids quadratic worst cases on sorted grids.
        a = pts[order[lo], axis]
        b = pts[order[mid], axis]
        c = pts[order[hi], axis]
        if a > b:
            t = order[lo]
            order[lo] = order[mid]
            order[mid] = t
            a, b = b, a
        if b > c:
            t = order[mid]
            order[mid] = order[hi]
            order[hi] = t
            b, c = c, b
            if a > b:
                t = order[lo]
                order[lo] = order[mid]
                order[mid] = t
                a, b = b, a
        pivot = b
        i = lo
        j = hi
        while True:
            while pts[order[i], axis] < pivot:
                i += 1
            while pts[order[j], axis] > pivot:
                j -= 1
            if i >= j:
                break
            t = order[i]
            order[i] = order[j]
            order[j] = t
            i += 1
            j -= 1
        if j < kth:
            lo = j + 1
        else:
            hi = j


@njit(cache=True)
def _build_kdtree_kernel(
    pts: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build an exact balanced median-split KD-tree (height <= ceil(log2(N/16))+1)."""
    n = pts.shape[0]
    dim = pts.shape[1]
    order = np.arange(n, dtype=np.int64)
    # Each leaf holds at least _LEAF_SIZE // 2 points (median split when size > _LEAF_SIZE),
    # so a binary tree has at most 2 * ceil(n / (_LEAF_SIZE // 2)) + 1 nodes.
    min_leaf = max(_LEAF_SIZE // 2, 1)
    cap = max(2 * ((n + min_leaf - 1) // min_leaf) + 8, 8)
    split_dim = np.full(cap, -1, dtype=np.int32)
    split_val = np.zeros(cap, dtype=np.float64)
    left = np.full(cap, -1, dtype=np.int64)
    right = np.full(cap, -1, dtype=np.int64)
    leaf_start = np.zeros(cap, dtype=np.int64)
    leaf_end = np.zeros(cap, dtype=np.int64)

    st_node = np.empty(64, dtype=np.int64)
    st_start = np.empty(64, dtype=np.int64)
    st_end = np.empty(64, dtype=np.int64)
    st_node[0] = 0
    st_start[0] = 0
    st_end[0] = n
    top = 1
    next_id = 1

    while top > 0:
        top -= 1
        nid = st_node[top]
        s = st_start[top]
        e = st_end[top]
        if e - s <= _LEAF_SIZE:
            split_dim[nid] = -1
            leaf_start[nid] = s
            leaf_end[nid] = e
            continue

        best_d = 0
        best_span = -1.0
        for d in range(dim):
            lo = pts[order[s], d]
            hi = lo
            for k in range(s + 1, e):
                v = pts[order[k], d]
                if v < lo:
                    lo = v
                elif v > hi:
                    hi = v
            span = hi - lo
            if span > best_span:
                best_span = span
                best_d = d

        if best_span <= 0.0:
            split_dim[nid] = -1
            leaf_start[nid] = s
            leaf_end[nid] = e
            continue

        mid_idx = (s + e) >> 1
        _quickselect_axis(pts, order, s, e - 1, mid_idx, best_d)
        mid_val = pts[order[mid_idx], best_d]

        l_id = next_id
        r_id = next_id + 1
        next_id += 2
        split_dim[nid] = best_d
        split_val[nid] = mid_val
        left[nid] = l_id
        right[nid] = r_id

        st_node[top] = r_id
        st_start[top] = mid_idx
        st_end[top] = e
        top += 1
        st_node[top] = l_id
        st_start[top] = s
        st_end[top] = mid_idx
        top += 1

    return (
        order,
        split_dim[:next_id],
        split_val[:next_id],
        left[:next_id],
        right[:next_id],
        leaf_start[:next_id],
        leaf_end[:next_id],
    )


@njit(cache=True, parallel=True)
def _query_kdtree_kernel(
    ref_pts: np.ndarray,
    query_pts: np.ndarray,
    order: np.ndarray,
    split_dim: np.ndarray,
    split_val: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    leaf_start: np.ndarray,
    leaf_end: np.ndarray,
) -> np.ndarray:
    n_q = query_pts.shape[0]
    dim = ref_pts.shape[1]
    out = np.empty(n_q, dtype=np.int64)

    for q in prange(n_q):
        best_dist = np.inf
        best_idx = 0

        stack_node = np.empty(64, dtype=np.int64)
        stack_lb = np.empty(64, dtype=np.float64)
        sp = 1
        stack_node[0] = 0
        stack_lb[0] = 0.0

        while sp > 0:
            sp -= 1
            if stack_lb[sp] > best_dist:
                continue
            nid = stack_node[sp]
            sd = split_dim[nid]
            if sd < 0:
                s = leaf_start[nid]
                e = leaf_end[nid]
                for k in range(s, e):
                    idx = order[k]
                    d2 = 0.0
                    for d in range(dim):
                        diff = query_pts[q, d] - ref_pts[idx, d]
                        d2 += diff * diff
                    if d2 < best_dist or (d2 == best_dist and idx < best_idx):
                        best_dist = d2
                        best_idx = idx
            else:
                diff = query_pts[q, sd] - split_val[nid]
                if diff <= 0.0:
                    near_child = left[nid]
                    far_child = right[nid]
                else:
                    near_child = right[nid]
                    far_child = left[nid]
                far_lb = diff * diff
                if far_lb <= best_dist:
                    stack_node[sp] = far_child
                    stack_lb[sp] = far_lb
                    sp += 1
                stack_node[sp] = near_child
                stack_lb[sp] = 0.0
                sp += 1

        out[q] = best_idx
    return out


def nearest_points(ref_pts: np.ndarray, query_pts: np.ndarray) -> np.ndarray:
    """Index into ``ref_pts`` of the exact Euclidean nearest neighbor for every row of ``query_pts``."""
    ref = np.ascontiguousarray(ref_pts, dtype=np.float64)
    qry = np.ascontiguousarray(query_pts, dtype=np.float64)
    if ref.shape[0] == 0:
        raise ValueError("reference point set must be non-empty")
    if qry.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    tree = _build_kdtree_kernel(ref)
    return _query_kdtree_kernel(ref, qry, *tree)
