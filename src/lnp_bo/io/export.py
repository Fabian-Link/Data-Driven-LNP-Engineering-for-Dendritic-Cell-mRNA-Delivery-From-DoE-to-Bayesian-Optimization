# src/lnp_bo/io/export.py
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
    from torch import Tensor
except Exception:  # pragma: no cover
    torch = None
    Tensor = object  # type: ignore

try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_LIPID_COLS: Tuple[str, str, str, str] = (
    "PEG-DMG-2000",
    "Ionizable Lipid",
    "DOPE",
    "Cholesterol",
)
DEFAULT_NP_COL = "N/P"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _now_iso(tz: str = "Europe/Berlin") -> str:
    if ZoneInfo is not None:
        return datetime.now(ZoneInfo(tz)).strftime("%Y%m%d_%H%M%S")
    # fallback: naive local time
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _to_numpy(a: Any) -> np.ndarray:
    if torch is not None and isinstance(a, Tensor):
        a = a.detach().cpu().numpy()
    return np.asarray(a)


def timestamped_path(
    out_dir: Union[str, Path],
    stem: str,
    *,
    ext: str = "xlsx",
    tz: str = "Europe/Berlin",
) -> Path:
    """
    Build `<out_dir>/<stem>_<YYYYmmdd_HHMMSS>.<ext>` and ensure the directory exists.
    """
    _ensure_dir(out_dir)
    ts = _now_iso(tz)
    return Path(out_dir) / f"{stem}_{ts}.{ext.lstrip('.')}"


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

def build_design_df(
    mix_simplex: Union[np.ndarray, Tensor],
    *,
    np_vals: Optional[Union[np.ndarray, Tensor]] = None,
    mix_cols: Sequence[str] = DEFAULT_LIPID_COLS,
    np_col: str = DEFAULT_NP_COL,
    round_cols: Optional[int] = None,
) -> pd.DataFrame:
    """
    Create a tidy DataFrame for a design:
      columns: mix_cols (4) + optional np_col (1)
    """
    X = _to_numpy(mix_simplex)
    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError("mix_simplex must be shape (N, 4).")

    data = {c: X[:, i] for i, c in enumerate(mix_cols)}
    if np_vals is not None:
        v = _to_numpy(np_vals).reshape(-1)
        if v.shape[0] != X.shape[0]:
            raise ValueError("Length of np_vals must match number of rows in mix_simplex.")
        data[np_col] = v

    df = pd.DataFrame(data)
    if round_cols is not None:
        df = df.round(round_cols)
    return df


def build_alr_np_df(
    z_alr_np: Union[np.ndarray, Tensor],
    *,
    np_col: str = DEFAULT_NP_COL,
    round_cols: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convert ALR+N/P (K columns: first K-1 are ALR, last is N/P) to a DataFrame
    with columns ['alr_1','alr_2','alr_3', np_col].
    """
    Z = _to_numpy(z_alr_np)
    if Z.ndim != 2 or Z.shape[1] < 2:
        raise ValueError("z_alr_np must be 2D with at least 2 columns (ALR dims + N/P).")
    k = Z.shape[1]
    cols = [f"alr_{i+1}" for i in range(k - 1)] + [np_col]
    df = pd.DataFrame(Z, columns=cols)
    if round_cols is not None:
        df = df.round(round_cols)
    return df


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def save_csv(df: pd.DataFrame, path: Union[str, Path], *, float_format: Optional[str] = None) -> Path:
    path = Path(path)
    _ensure_dir(path.parent)
    df.to_csv(path, index=False, float_format=float_format)
    return path


def save_excel(
    sheets: Mapping[str, pd.DataFrame],
    path: Union[str, Path],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """
    Save multiple sheets to an .xlsx workbook.
      - `sheets`: mapping from sheet name to DataFrame
      - `metadata`: optional mapping; if provided, a 'meta' sheet is added
    """
    path = Path(path)
    _ensure_dir(path.parent)

    # Pandas will choose an engine (prefers openpyxl for .xlsx)
    with pd.ExcelWriter(path) as writer:
        for name, df in sheets.items():
            # sanitize sheet name (Excel limit 31 chars)
            sname = str(name)[:31] if name else "Sheet1"
            df.to_excel(writer, index=False, sheet_name=sname)
        if metadata:
            # flatten simple dicts; nested objects become JSON
            meta_rows = []
            for k, v in metadata.items():
                if is_dataclass(v):
                    v = asdict(v)
                if isinstance(v, (dict, list, tuple)):
                    try:
                        v = json.dumps(v, ensure_ascii=False, indent=2)
                    except Exception:
                        v = str(v)
                meta_rows.append({"key": k, "value": v})
            meta_df = pd.DataFrame(meta_rows)
            meta_df.to_excel(writer, index=False, sheet_name="meta")
    return path


# ---------------------------------------------------------------------------
# High-level convenience for ProposedBatch
# ---------------------------------------------------------------------------

def export_proposed_batch(
    batch: Any,
    out_dir: Union[str, Path],
    *,
    mix_cols: Sequence[str] = DEFAULT_LIPID_COLS,
    np_col: str = DEFAULT_NP_COL,
    stem: str = "candidates",
    tz: str = "Europe/Berlin",
    also_csv: bool = False,
    round_cols: Optional[int] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Path]:
    """
    Export a batch of proposed candidates to a timestamped Excel (and optional CSV).

    The function is dataclass/duck-typed:
      - If `batch` has attributes like `mix_simplex`, `np_column`, `z_alr_np`, `acq_value`,
        they are used (as in models.acquisition.ProposedBatch).
      - Otherwise, `batch` may be a dict with the same keys.

    Returns
    -------
    dict with saved file paths (keys: 'excel', optionally 'csv_5d').
    """
    # Extract fields in a tolerant way
    def _get(obj, key, default=None):
        return getattr(obj, key, obj.get(key, default) if isinstance(obj, Mapping) else default)

    mix = _get(batch, "mix_simplex")
    npr = _get(batch, "np_column")
    z = _get(batch, "z_alr_np")
    acq_val = _get(batch, "acq_value")

    if mix is None or npr is None:
        raise ValueError("`batch` must contain 'mix_simplex' and 'np_column'.")

    df_5d = build_design_df(mix, np_vals=npr, mix_cols=mix_cols, np_col=np_col, round_cols=round_cols)
    sheets: Dict[str, pd.DataFrame] = {"candidates_5d": df_5d}

    if z is not None:
        df_z = build_alr_np_df(z, np_col=np_col, round_cols=round_cols)
        sheets["alr_np"] = df_z

    # metadata
    meta: Dict[str, Any] = {"export_time": _now_iso(tz)}
    if acq_val is not None:
        try:
            meta["acq_value"] = float(_to_numpy(acq_val).reshape(-1)[0])
        except Exception:
            meta["acq_value"] = str(acq_val)
    if extra_metadata:
        meta.update(extra_metadata)

    # Save Excel
    excel_path = timestamped_path(out_dir, stem=stem, ext="xlsx", tz=tz)
    save_excel(sheets, excel_path, metadata=meta)

    out: Dict[str, Path] = {"excel": excel_path}

    # Optional CSV for just the main table
    if also_csv:
        csv_path = timestamped_path(out_dir, stem=stem, ext="csv", tz=tz)
        save_csv(df_5d, csv_path, float_format="%.6g" if round_cols is None else None)
        out["csv_5d"] = csv_path

    return out


# ---------------------------------------------------------------------------
# Simple one-shot helpers
# ---------------------------------------------------------------------------

def export_design_table(
    table: Union[pd.DataFrame, np.ndarray, Tensor],
    path: Union[str, Path],
    *,
    header: Optional[Sequence[str]] = None,
    csv_float_format: Optional[str] = "%.6g",
) -> Path:
    """
    Save a single table to CSV or Excel, inferred from `path` suffix.
    - If `table` is an array/tensor, `header` is required.
    """
    path = Path(path)
    if isinstance(table, pd.DataFrame):
        df = table
    else:
        if header is None:
            raise ValueError("`header` must be provided when passing an array/tensor.")
        arr = _to_numpy(table)
        if arr.ndim != 2 or arr.shape[1] != len(header):
            raise ValueError("`table` must be (N, len(header)).")
        df = pd.DataFrame(arr, columns=list(header))

    if path.suffix.lower() == ".csv":
        return save_csv(df, path, float_format=csv_float_format)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        return save_excel({"table": df}, path)
    else:
        raise ValueError(f"Unsupported file extension for {path}. Use .csv or .xlsx.")
