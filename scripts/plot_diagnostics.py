#!/usr/bin/env python3
# scripts/plot_diagnostics.py
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import yaml
import torch
import matplotlib.pyplot as plt

from lnp_bo.io.load import (
    prepare_dataset,
    to_torch_tensors,
    compute_noise_prior_from_triplicates,
)
from lnp_bo.models.gp import initialize_model
from lnp_bo.viz.gp_diagnostics import (
    plot_loss_curve,
    plot_hyperparam_trajectories,
    plot_parity,
    plot_predictive_variance_hist,
    plot_variance_vs_distance,
    loo_summary,
)


# ---------------------------
# YAML utils
# ---------------------------

def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------
# History loader
# ---------------------------

def _load_history(path: Optional[str | Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[warn] history file not found: {p}")
        return None
    try:
        if p.suffix.lower() in {".json"}:
            return json.loads(Path(p).read_text(encoding="utf-8"))
        if p.suffix.lower() in {".pkl", ".pickle"}:
            return pickle.loads(Path(p).read_bytes())
        if p.suffix.lower() == ".npz":
            import numpy as np
            npz = np.load(p, allow_pickle=True)
            return {k: npz[k].tolist() for k in npz.files}
        # naive: try json first, else pickle
        try:
            return json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            return pickle.loads(Path(p).read_bytes())
    except Exception as e:
        print(f"[warn] failed to load history from {p}: {e}")
        return None


# ---------------------------
# CLI
# ---------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot GP diagnostics (parity, variance, training curves)."
    )
    p.add_argument("--paths", default="configs/paths.yaml", help="Path to paths YAML (data/state/output).")
    p.add_argument("--state", default=None, help="Override path to trained GP state_dict (.pt/.pth).")
    p.add_argument("--history", default=None, help="Optional training history file (.json/.pkl/.npz).")
    p.add_argument("--outdir", default=None, help="Directory to save figures (defaults to results/figures from YAML).")
    p.add_argument("--prefix", default="diag", help="Filename prefix for saved figures.")
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Figure format.")
    p.add_argument("--dpi", type=int, default=150, help="DPI for raster formats.")
    p.add_argument("--show", action="store_true", help="Also display figures interactively.")
    return p


# ---------------------------
# Save helper
# ---------------------------

def _save(fig, out_dir: Path, name: str, fmt: str, dpi: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# ---------------------------
# Main
# ---------------------------

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg_paths = _load_yaml(args.paths)

    data_xlsx  = cfg_paths.get("data", {}).get("raw", "data/raw/LNP_Formulation_All_Data_Mean.xlsx")
    trips_xlsx = cfg_paths.get("data", {}).get("triplicates", "data/raw/LNP_Formulation_8_Luciferase_triplicates.xlsx")
    trips_name = cfg_paths.get("data", {}).get("triplicates_name_col", "Name")
    resp_col   = cfg_paths.get("data", {}).get("response_col", "Fluoreszenz [RLU]")

    state_file = args.state or cfg_paths.get("models", {}).get("state_dict", "results/models/gp_state.pt")
    outdir     = args.outdir or cfg_paths.get("outputs", {}).get("figures_dir", "results/figures")

    out_dir = Path(outdir)
    state_path = Path(state_file)
    if not state_path.exists():
        print(f"[error] state_dict not found at: {state_path}")
        return 2

    # ---- load dataset (for NEI baseline dtype/device & plots)
    dataset = prepare_dataset(
        data_xlsx,
        lipid_cols=("PEG-DMG-2000","Ionizable Lipid","DOPE","Cholesterol"),
        np_col="N/P",
        response_col=resp_col,
    )
    train_x, train_y = to_torch_tensors(dataset.X_aug, dataset.y_log, device="cpu", dtype=torch.double)

    # ---- reconstruct model with empirical noise prior, then load weights
    _, _, noise_prior = compute_noise_prior_from_triplicates(
        trips_xlsx, name_col=trips_name, response_col=resp_col
    )
    mll, model = initialize_model(train_x, train_y, noise_prior=noise_prior)

    sd = torch.load(state_path, map_location="cpu")
    model.load_state_dict(sd)
    model.eval(); model.likelihood.eval()

    # ---- 1) Training curves (optional)
    history = _load_history(args.history)
    if history is not None and isinstance(history, dict) and history.get("loss"):
        fig, ax = plot_loss_curve(history)
        p = _save(fig, out_dir, f"{args.prefix}_loss_curve", args.format, args.dpi)
        print(f"Saved: {p}")

        fig, axes = plot_hyperparam_trajectories(history)
        p = _save(fig, out_dir, f"{args.prefix}_hyperparams", args.format, args.dpi)
        print(f"Saved: {p}")
    else:
        if args.history:
            print("[warn] no valid history found or 'loss' missing; skipping training curves.")

    # ---- 2) Parity
    fig, ax, r2 = plot_parity(model, train_x, train_y, title="GP Parity (Train Data)")
    p = _save(fig, out_dir, f"{args.prefix}_parity", args.format, args.dpi)
    print(f"Saved: {p}  (R^2={r2:.3f})")

    # ---- 3) Predictive variance diagnostics
    fig, ax = plot_predictive_variance_hist(model, num_samples=2000, bins=40)
    p = _save(fig, out_dir, f"{args.prefix}_var_hist", args.format, args.dpi)
    print(f"Saved: {p}")

    fig, ax = plot_variance_vs_distance(model, train_x, num_random=1000)
    p = _save(fig, out_dir, f"{args.prefix}_var_vs_dist", args.format, args.dpi)
    print(f"Saved: {p}")

    # ---- 4) LOO summary
    rep = loo_summary(model, train_x, train_y, k_sigma=2.0)
    print("\nLOO summary:")
    for k, v in rep.items():
        print(f"  {k}: {v:.4f}")

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
