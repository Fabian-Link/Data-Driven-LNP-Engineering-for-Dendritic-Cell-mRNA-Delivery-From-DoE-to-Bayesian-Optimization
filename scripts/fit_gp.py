#!/usr/bin/env python3
# scripts/fit_gp.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import numpy as np
import torch
from torch.optim import Adam

from botorch.optim.fit import fit_gpytorch_mll_torch, fit_gpytorch_mll_scipy
from botorch.optim.core import OptimizationResult

from lnp_bo.utils.seed import set_global_seed
from lnp_bo.io.load import (
    prepare_dataset,
    to_torch_tensors,
    compute_noise_prior_from_triplicates,
)
from lnp_bo.models.gp import initialize_model  # uses Normalize + Standardize
from lnp_bo.viz.gp_diagnostics import (
    plot_loss_curve,
    plot_hyperparam_trajectories,
)

# ---------------------------
# YAML utils
# ---------------------------

def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------
# I/O helpers
# ---------------------------

def _ensure_dir(p: str | Path) -> Path:
    pp = Path(p)
    pp.mkdir(parents=True, exist_ok=True)
    return pp

def _save_json(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    _ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return p


# ---------------------------
# CLI
# ---------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train GP model for LNP/DC mRNA transfection data and save state_dict + history."
    )
    p.add_argument("--paths", default="configs/paths.yaml", help="Path to paths YAML.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--device", default="cpu", help="torch device (e.g., cpu, cuda).")
    p.add_argument("--dtype", default="double", choices=["float", "double"], help="torch dtype.")
    p.add_argument("--lr", type=float, default=1e-2, help="Adam learning rate.")
    p.add_argument("--steps", type=int, default=800, help="Training steps for Adam fit.")
    p.add_argument("--optimizer", choices=["adam", "lbfgs"], default="adam", help="Optimizer backend.")
    p.add_argument("--maxiter", type=int, default=200, help="L-BFGS-B max iterations (if --optimizer lbfgs).")
    p.add_argument("--save-dir", default=None, help="Override output dir for models/history (defaults from YAML).")
    p.add_argument("--tag", default=None, help="Optional run tag to suffix output filenames.")
    p.add_argument("--figs", action="store_true", help="Also save loss + hyperparameter trajectory figures.")
    return p


# ---------------------------
# Main
# ---------------------------

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # ---- load config
    cfg_paths = _load_yaml(args.paths)
    data_xlsx  = cfg_paths.get("data", {}).get("raw", "data/raw/LNP_Formulation_All_Data_Mean.xlsx")
    trips_xlsx = cfg_paths.get("data", {}).get("triplicates", "data/raw/LNP_Formulation_8_Luciferase_triplicates.xlsx")
    trips_name = cfg_paths.get("data", {}).get("triplicates_name_col", "Name")
    resp_col   = cfg_paths.get("data", {}).get("response_col", "Fluoreszenz [RLU]")

    # output dirs / files
    models_dir = Path(cfg_paths.get("outputs", {}).get("models_dir", "results/models"))
    figures_dir = Path(cfg_paths.get("outputs", {}).get("figures_dir", "results/figures"))
    if args.save_dir:
        models_dir = Path(args.save_dir)

    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tag = f"_{args.tag}" if args.tag else ""

    state_path  = models_dir / f"gp_state{tag}.pt"
    history_path= models_dir / f"history{tag}.json"

    # ---- torch dtype / device
    dtype = torch.double if args.dtype == "double" else torch.float
    device = torch.device(args.device)

    # ---- seed
    set_global_seed(args.seed, deterministic=False)

    # ---- dataset
    dataset = prepare_dataset(
        data_xlsx,
        lipid_cols=("PEG-DMG-2000","Ionizable Lipid","DOPE","Cholesterol"),
        np_col="N/P",
        response_col=resp_col,
    )
    train_x, train_y = to_torch_tensors(dataset.X_aug, dataset.y_log, device=device, dtype=dtype)

    # ---- empirical noise prior
    _, _, noise_prior = compute_noise_prior_from_triplicates(
        trips_xlsx, name_col=trips_name, response_col=resp_col
    )

    # ---- build model
    mll, model = initialize_model(train_x, train_y, noise_prior=noise_prior)
    model.train(); mll.train()

    # ---- training history
    history: Dict[str, list[float]] = {
        "loss": [],
        "lengthscale": [],
        "outputscale": [],
        "noise": [],
    }

    def _callback(params: Dict[str, torch.Tensor], result: OptimizationResult):
        # negative log marginal likelihood
        history["loss"].append(float(result.fval))
        # read constrained hyperparameters
        ls = float(model.covar_module.base_kernel.lengthscale.mean().item())
        os = float(model.covar_module.outputscale.item())
        nv = float(model.likelihood.noise.item())
        history["lengthscale"].append(ls)
        history["outputscale"].append(os)
        history["noise"].append(nv)

    # ---- fit
    if args.optimizer == "adam":
        fit_gpytorch_mll_torch(
            mll,
            optimizer=lambda params: Adam(params, lr=args.lr),
            step_limit=int(args.steps),
            callback=_callback,
        )
    else:
        # L-BFGS-B path
        fit_gpytorch_mll_scipy(
            mll,
            method="L-BFGS-B",
            options={"maxiter": int(args.maxiter)},
            callback=_callback,
        )

    # ---- save weights
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), state_path)
    print(f"Saved state_dict → {state_path}")

    # ---- save history
    _save_json(history, history_path)
    print(f"Saved history     → {history_path}")

    # ---- optional figures
    if args.figs:
        fig, _ = plot_loss_curve(history)
        p = figures_dir / f"train_loss{tag}.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        print(f"Saved figure      → {p}")

        fig, _ = plot_hyperparam_trajectories(history)
        p = figures_dir / f"train_hparams{tag}.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        print(f"Saved figure      → {p}")

    # ---- quick summary
    ls = float(model.covar_module.base_kernel.lengthscale.mean().item())
    os = float(model.covar_module.outputscale.item())
    nv = float(model.likelihood.noise.item())
    print("\nLearned hyperparameters:")
    print(f"  mean ARD lengthscale : {ls:.4g}")
    print(f"  outputscale (signal) : {os:.4g}")
    print(f"  noise variance       : {nv:.4g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
