#!/usr/bin/env python3
# scripts/propose_candidates.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml
import numpy as np
import torch

from lnp_bo.utils.seed import set_global_seed
from lnp_bo.io.load import (
    prepare_dataset,
    to_torch_tensors,
    compute_noise_prior_from_triplicates,
)
from lnp_bo.models.gp import initialize_model
from lnp_bo.models.bounds import compute_alr_bounds_from_simplex_bounds
from lnp_bo.models.acquisition import propose_batch
from lnp_bo.io.export import export_proposed_batch


# ---------------------------------------------------------------------
# YAML utils
# ---------------------------------------------------------------------

def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _as_path(x: str | Path | None) -> Path | None:
    return None if x is None else Path(x)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Propose next LNP candidates via Bayesian optimization (ALR+N/P)."
    )
    p.add_argument("--bounds", default="configs/bounds.yaml", help="Path to bounds YAML.")
    p.add_argument("--acq", default="configs/acquisition.yaml", help="Path to acquisition YAML.")
    p.add_argument("--paths", default="configs/paths.yaml", help="Path to paths YAML.")

    # quick overrides
    p.add_argument("--acq-name", type=str, default=None, help="Override acquisition name (e.g., LogNEI, LogEI, UCB).")
    p.add_argument("-q", "--batch-size", type=int, default=None, help="Override batch size.")
    p.add_argument("--restarts", type=int, default=None, help="num_restarts for optimize_acqf.")
    p.add_argument("--raw-samples", type=int, default=None, help="raw_samples for optimize_acqf.")
    p.add_argument("--mc-samples", type=int, default=None, help="MC samples for MC acquisitions.")
    p.add_argument("--beta-ucb", type=float, default=None, help="beta for UCB.")
    p.add_argument("--seed", type=int, default=None, help="Random seed.")

    p.add_argument("--state", type=str, default=None, help="Path to trained GP state_dict (.pt/.pth).")
    p.add_argument("--out-dir", type=str, default=None, help="Directory to write candidates (default from paths.yaml).")
    p.add_argument("--csv", action="store_true", help="Also write a CSV next to the Excel.")
    p.add_argument("--round", type=int, default=None, help="Round saved numeric columns to N decimals.")

    return p


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # ---- load configs
    cfg_bounds = _load_yaml(args.bounds)
    cfg_acq    = _load_yaml(args.acq)
    cfg_paths  = _load_yaml(args.paths)

    # Bounds YAML schema
    labels = tuple(cfg_bounds.get("mixture", {}).get("labels", ["PEG-DMG-2000","Ionizable Lipid","DOPE","Cholesterol"]))
    L = np.array(cfg_bounds.get("mixture", {}).get("lower", [0.01, 0.2, 0.05, 0.05]), dtype=float)
    U = np.array(cfg_bounds.get("mixture", {}).get("upper", [0.10, 0.8, 0.50, 0.80]), dtype=float)
    npr_lower = float(cfg_bounds.get("n_p", {}).get("lower", 1.0))
    npr_upper = float(cfg_bounds.get("n_p", {}).get("upper", 12.0))

    # Acquisition YAML schema
    acq_name    = args.acq_name or cfg_acq.get("type", "LogNEI")
    q           = args.batch_size or int(cfg_acq.get("q", 3))
    num_restarts= args.restarts or int(cfg_acq.get("restarts", 10))
    raw_samples = args.raw_samples or int(cfg_acq.get("raw_samples", 512))
    mc_samples  = args.mc_samples or int(cfg_acq.get("mc_samples", 256))
    beta_ucb    = args.beta_ucb if args.beta_ucb is not None else float(cfg_acq.get("beta_ucb", 0.05))
    seed        = args.seed or int(cfg_acq.get("seed", 42))

    # Paths YAML schema
    data_xlsx   = cfg_paths.get("data", {}).get("raw", "data/raw/LNP_Formulation_All_Data_Mean.xlsx")
    trips_xlsx  = cfg_paths.get("data", {}).get("triplicates", "data/raw/LNP_Formulation_8_Luciferase_triplicates.xlsx")
    trips_name  = cfg_paths.get("data", {}).get("triplicates_name_col", "Name")
    resp_col    = cfg_paths.get("data", {}).get("response_col", "Fluoreszenz [RLU]")
    state_file  = args.state or cfg_paths.get("models", {}).get("state_dict", "results/models/gp_state.pt")
    out_dir     = args.out_dir or cfg_paths.get("outputs", {}).get("candidates_dir", "results/candidates")

    # ---- seed
    set_global_seed(seed, deterministic=False)

    # ---- prepare dataset (rebuild train_x, train_y to serve as baseline for NEI/LogNEI)
    dataset = prepare_dataset(
        data_xlsx,
        lipid_cols=labels,
        np_col="N/P",
        response_col=resp_col,
    )
    train_x, train_y = to_torch_tensors(dataset.X_aug, dataset.y_log, device="cpu", dtype=torch.double)

    # ---- rebuild model skeleton and load weights
    #     We need a likelihood to exist before loading; reuse your empirical noise prior.
    _, _, noise_prior = compute_noise_prior_from_triplicates(
        trips_xlsx, name_col=trips_name, response_col=resp_col
    )
    mll, model = initialize_model(train_x, train_y, noise_prior=noise_prior)

    state_path = Path(state_file)
    if not state_path.exists():
        print(f"[error] state_dict not found at: {state_path}", file=sys.stderr)
        return 2
    sd = torch.load(state_path, map_location="cpu")
    try:
        model.load_state_dict(sd)
    except Exception as e:
        print(f"[error] failed to load state_dict from {state_path}: {e}", file=sys.stderr)
        return 3
    model.eval(); model.likelihood.eval()

    # ---- build ALR+N/P bounds
    simplex_lb = torch.tensor(L, dtype=train_x.dtype, device=train_x.device)
    simplex_ub = torch.tensor(U, dtype=train_x.dtype, device=train_x.device)

    alr_lb, alr_ub = compute_alr_bounds_from_simplex_bounds(
        simplex_lb, simplex_ub, num_samples=int(cfg_acq.get("alr_bounds_samples", 20000))
    )
    bounds_alr_np = torch.stack(
        [
            torch.cat([alr_lb, torch.tensor([npr_lower], dtype=train_x.dtype)]),
            torch.cat([alr_ub, torch.tensor([npr_upper], dtype=train_x.dtype)]),
        ],
        dim=0,
    )

    # ---- propose a batch
    batch = propose_batch(
        model=model,
        train_x=train_x,
        train_y=train_y,
        bounds_alr_np=bounds_alr_np,
        simplex_lb=simplex_lb,
        simplex_ub=simplex_ub,
        acq_name=acq_name,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        mc_samples=mc_samples,
        beta_ucb=beta_ucb,
        seed=seed,
    )

    # ---- pretty print to console
    cand = batch.candidates_5d.detach().cpu().numpy()
    print(f"\nProposed candidates ({q}): [ {', '.join(labels)} | N/P ]")
    for i, row in enumerate(cand, 1):
        mix_str = "  ".join(f"{lab}: {val:0.4f}" for lab, val in zip(labels, row[:4]))
        print(f"  {i:02d}) {mix_str}  |  N/P: {row[4]:0.3f}")

    # ---- export
    paths = export_proposed_batch(
        batch,
        out_dir=out_dir,
        mix_cols=labels,
        np_col="N/P",
        stem=f"bo_{acq_name.lower()}_q{q}",
        also_csv=bool(args.csv),
        round_cols=args.round,
        extra_metadata={
            "acq": acq_name,
            "q": q,
            "num_restarts": num_restarts,
            "raw_samples": raw_samples,
            "mc_samples": mc_samples,
            "beta_ucb": beta_ucb,
            "seed": seed,
            "state_file": str(state_path),
            "bounds_yaml": str(Path(args.bounds).resolve()),
            "acq_yaml": str(Path(args.acq).resolve()),
            "paths_yaml": str(Path(args.paths).resolve()),
        },
    )
    print(f"\nSaved: {paths['excel']}")
    if "csv_5d" in paths:
        print(f"Saved: {paths['csv_5d']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
