#!/usr/bin/env python3
# scripts/export_excel.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
import numpy as np
import pandas as pd

from lnp_bo.io.export import (
    build_design_df,
    save_excel,
    save_csv,
    timestamped_path,
)
from lnp_bo.design.sampling import generate_1d_sobol


# ---------------------------
# YAML helpers
# ---------------------------

def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------
# I/O + detection helpers
# ---------------------------

def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    if p.suffix.lower() in {".xlsx", ".xls"}:
        # read first sheet by default
        return pd.read_excel(p)
    elif p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported input format: {p.suffix} (use .csv/.xlsx)")

def _pick_mixture_columns(df: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    """
    Try to select mixture columns in a robust way:
    - Prefer exact label order if all present
    - Else accept common fallbacks like x1..x4
    - Else, take the first 4 numeric columns
    """
    # exact labels
    if all(c in df.columns for c in labels):
        return df.loc[:, list(labels)]

    # common fallbacks
    fallbacks = [
        ["x1", "x2", "x3", "x4"],
        ["X1", "X2", "X3", "X4"],
        [0, 1, 2, 3],  # if RangeIndex columns
    ]
    for cols in fallbacks:
        if all(c in df.columns for c in cols):
            return df.loc[:, cols].rename(columns=dict(zip(cols, labels)))

    # otherwise: first 4 numeric columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 4:
        picked = numeric_cols[:4]
        out = df.loc[:, picked].copy()
        out.columns = list(labels)
        return out

    raise KeyError(
        "Could not infer 4 mixture columns. "
        f"Expected labels {list(labels)} or columns like x1..x4."
    )

def _extract_np_column(
    df: pd.DataFrame,
    *,
    np_col_name: str = "N/P",
    np_source: str = "auto",   # 'auto' | 'file' | 'generate'
    np_file: Optional[str] = None,
    n_rows: Optional[int] = None,
    np_lower: float = 1.0,
    np_upper: float = 12.0,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Retrieve or generate the N/P column.
    - auto: use df[np_col_name] if present and fully numeric; else generate Sobol
    - file: read a 1-col CSV/XLSX; column name may be np_col_name or any numeric single column
    - generate: Sobol 1D in [np_lower, np_upper] with length n_rows
    """
    np_col_name = str(np_col_name)

    if np_source not in {"auto", "file", "generate"}:
        raise ValueError("--np-source must be one of {'auto','file','generate'}")

    if np_source in {"auto", "file"}:
        # try to use from df (auto)
        if np_source == "auto" and np_col_name in df.columns:
            col = pd.to_numeric(df[np_col_name], errors="coerce")
            if not col.isna().any():
                return col.to_numpy(float)

        # try to read from a file
        if np_source == "file":
            if np_file is None:
                raise ValueError("--np-file is required when --np-source file")
            sub = _read_table(np_file)
            # prefer np_col_name if present, else the first numeric column
            if np_col_name in sub.columns:
                v = pd.to_numeric(sub[np_col_name], errors="coerce")
            else:
                num_cols = [c for c in sub.columns if pd.api.types.is_numeric_dtype(sub[c])]
                if not num_cols:
                    raise ValueError(f"No numeric column found in {np_file} for N/P.")
                v = pd.to_numeric(sub[num_cols[0]], errors="coerce")
            if v.isna().any():
                raise ValueError(f"N/P column from {np_file} contains NaNs after coercion.")
            return v.to_numpy(float)

    # fallback: generate
    if n_rows is None:
        raise ValueError("n_rows must be provided to generate N/P.")
    return generate_1d_sobol(n_rows, lower=float(np_lower), upper=float(np_upper), seed=seed)


def _renormalize_rows(mix: np.ndarray) -> np.ndarray:
    mix = np.asarray(mix, dtype=float)
    mix = np.clip(mix, 1e-12, None)
    mix = mix / mix.sum(axis=1, keepdims=True)
    return mix


# ---------------------------
# CLI
# ---------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Assemble a 5D design table (4-part mixture + N/P) and export to Excel/CSV."
    )
    p.add_argument("--input", required=True, help="Path to 4D mixture table (.csv/.xlsx).")
    p.add_argument("--bounds", default="configs/bounds.yaml", help="Bounds YAML (mixture labels + N/P limits).")
    p.add_argument("--paths", default="configs/paths.yaml", help="Paths YAML (for default output dir).")

    p.add_argument("--np-source", choices=["auto", "file", "generate"], default="auto",
                   help="Where to fetch N/P from (default: auto).")
    p.add_argument("--np-file", default=None, help="If --np-source file, path to 1-col table with N/P values.")
    p.add_argument("--seed", type=int, default=42, help="Seed for Sobol N/P generation.")

    p.add_argument("--renorm", action="store_true",
                   help="Renormalize each mixture row to sum=1 before export.")
    p.add_argument("--round", type=int, default=None, help="Round numeric columns to N decimals.")

    p.add_argument("--out-dir", default=None, help="Output directory (default: outputs.tables_dir in paths.yaml).")
    p.add_argument("--stem", default="design_5d", help="File stem for saved outputs.")
    p.add_argument("--csv", action="store_true", help="Also write a CSV alongside the Excel.")
    return p


# ---------------------------
# Main
# ---------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # -- load YAMLs
    cfg_bounds = _load_yaml(args.bounds)
    cfg_paths = _load_yaml(args.paths)

    labels: Tuple[str, str, str, str] = tuple(
        cfg_bounds.get("mixture", {}).get(
            "labels", ["PEG-DMG-2000", "Ionizable Lipid", "DOPE", "Cholesterol"]
        )
    )  # type: ignore[assignment]
    np_lower = float(cfg_bounds.get("n_p", {}).get("lower", 1.0))
    np_upper = float(cfg_bounds.get("n_p", {}).get("upper", 12.0))
    np_name = str(cfg_bounds.get("n_p", {}).get("label", "N/P"))

    out_dir = args.out_dir or cfg_paths.get("outputs", {}).get("tables_dir", "results/tables")
    out_dir = str(out_dir)

    # -- read input mixture table
    df_in = _read_table(args.input)
    df_mix = _pick_mixture_columns(df_in, labels)

    # -- optionally renormalize to the simplex
    mix = df_mix.to_numpy(float)
    if args.renorm:
        mix = _renormalize_rows(mix)

    # -- N/P column
    np_vec = _extract_np_column(
        df_in,
        np_col_name=np_name,
        np_source=args.np_source,
        np_file=args.np_file,
        n_rows=mix.shape[0],
        np_lower=np_lower,
        np_upper=np_upper,
        seed=args.seed,
    )

    # -- build tidy 5D DataFrame
    df_5d = build_design_df(
        mix,
        np_vals=np_vec,
        mix_cols=labels,
        np_col=np_name,
        round_cols=args.round,
    )

    # -- metadata sheet
    meta = {
        "source_input": str(Path(args.input).resolve()),
        "np_source": args.np_source if args.np_source != "file" else f"file:{Path(args.np_file).resolve() if args.np_file else ''}",
        "renormalized": bool(args.renorm),
        "n_rows": int(df_5d.shape[0]),
        "bounds_yaml": str(Path(args.bounds).resolve()),
        "paths_yaml": str(Path(args.paths).resolve()),
        "np_lower": np_lower,
        "np_upper": np_upper,
        "labels": list(labels),
    }

    # -- save Excel (+ optional CSV)
    xlsx_path = timestamped_path(out_dir, stem=args.stem, ext="xlsx")
    save_excel({"design_5d": df_5d}, xlsx_path, metadata=meta)
    print(f"Saved: {xlsx_path}")

    if args.csv:
        csv_path = timestamped_path(out_dir, stem=args.stem, ext="csv")
        save_csv(df_5d, csv_path, float_format=None if args.round is None else None)
        print(f"Saved: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
