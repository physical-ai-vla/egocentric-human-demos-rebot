"""Thin parquet helpers (pandas + pyarrow) so every stage writes the same way."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def write_parquet(path: Path, columns: Mapping[str, Sequence[Any] | np.ndarray]) -> Path:
    """Write a column dict; array-valued columns (e.g. landmarks) become list columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({k: _column(v) for k, v in columns.items()})
    frame.to_parquet(path, index=False)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def _column(values: Sequence[Any] | np.ndarray) -> Any:
    arr = np.asarray(values) if not isinstance(values, np.ndarray) else values
    if arr.dtype == object or arr.ndim <= 1:
        return list(values) if arr.dtype == object else arr
    # (N, ...) numeric -> list of flat arrays per row
    return [row.reshape(-1) for row in arr]


def stack_column(frame: pd.DataFrame, name: str, width: int | None = None) -> np.ndarray:
    """Turn a list column back into an (N, width) float array (NaN for missing rows)."""
    rows = frame[name].tolist()
    if width is None:
        width = next((len(r) for r in rows if r is not None and len(r) > 0), 0)
    out = np.full((len(rows), width), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        if r is not None and len(r) == width:
            out[i] = np.asarray(r, dtype=np.float64)
    return out


__all__ = ["read_parquet", "stack_column", "write_parquet"]
