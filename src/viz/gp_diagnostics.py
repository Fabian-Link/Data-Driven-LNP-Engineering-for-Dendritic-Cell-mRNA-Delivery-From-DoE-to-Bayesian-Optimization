from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from models.gp import distance_variance_profile, loo_predictive


# ---------------------------------
# Helpers
# ---------------------------------
def _finalize(fig, outpath: Optional[Path] = None, dpi: int = 300, show: bool = False):
    """Save/show/close a figure in a consistent way."""
    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight", transparent=True)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------
# Hyperparameter tracking plots
# ---------------------------------
def plot_training_loss(
    history: Dict[str, List[float]],
    outpath: Optional[Path] = None,
    show: bool = False,
):
    iters = list(range(len(history.get("loss", []))))
    fig = plt.figure(figsize=(6, 4))
    plt.plot(iters, history.get("loss", []), label="Neg. log-lik")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Marginal Likelihood Optimization")
    plt.legend()
    plt.tight_layout()
    _finalize(fig, outpath=outpath, show=show)


def plot_hyperparam_trajectories(
    history: Dict[str, List[float]],
    outpath: Optional[Path] = None,
    show: bool = False,
):
    iters = list(range(len(history.get("loss", []))))
    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    axes[0].plot(iters, history.get("lengthscale", []), label="mean ARD ℓ")
    axes[0].legend()
    axes[1].plot(iters, history.get("outputscale", []), label="outputscale")
    axes[1].legend()
    axes[2].plot(iters, history.get("noise", []), label="noise var")
    axes[2].legend()
    axes[2].set_xlabel("Iteration")
    plt.suptitle("Hyperparameter Trajectories (Adam)", y=1.02)
    plt.tight_layout()
    _finalize(fig, outpath=outpath, show=show)


# ---------------------------------
# Learned hyperparameters export
# ---------------------------------
def export_learned_hyperparameters(model, csv_path: Path) -> Dict[str, float]:
    """
    Save learned ARD lengthscales, outputscale (signal variance), and noise variance.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    lengthscale = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
    outputscale = float(model.covar_module.outputscale.item())
    noise = float(model.likelihood.noise.item())

    rows = [{"param": f"lengthscale_{i+1}", "value": float(v)} for i, v in enumerate(lengthscale)]
    rows += [
        {"param": "outputscale", "value": outputscale},
        {"param": "noise_variance", "value": noise},
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return {
        "outputscale": outputscale,
        "noise_variance": noise,
        "lengthscales": lengthscale.tolist(),
    }


# ---------------------------------
# Parity plot (+ R²)
# ---------------------------------
def parity_plot(
    model,
    X: torch.Tensor,
    Y: torch.Tensor,
    outpath: Optional[Path] = None,
    show: bool = False,
) -> float:
    """
    Generate a parity plot (pred vs true) and return R².
    """
    model.eval()
    model.likelihood.eval()
    with torch.no_grad():
        post = model.posterior(X)
        y_pred = post.mean.squeeze(-1).cpu().numpy()
        y_true = Y.squeeze(-1).cpu().numpy()

    # R² without sklearn
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")

    fig = plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, s=40, alpha=0.7)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "k--", linewidth=1)
    plt.xlabel("Actual log-Fluoreszenz")
    plt.ylabel("Predicted log-Fluoreszenz")
    plt.title(f"Parity Plot (R² = {r2:.3f})")
    plt.tight_layout()
    _finalize(fig, outpath=outpath, show=show)

    return r2


# ---------------------------------
# Variance vs. distance
# ---------------------------------
def variance_vs_distance_plot(
    model,
    train_x: torch.Tensor,
    outpath: Optional[Path] = None,
    points_csv: Optional[Path] = None,
    show: bool = False,
):
    """
    Scatter predictive variance vs distance-to-nearest-training-point in normalized space.

    Also optionally save the raw points to CSV.
    """
    d_train, var_train, d_rand, var_rand = distance_variance_profile(model, train_x)

    fig = plt.figure(figsize=(6, 4))
    plt.scatter(d_rand, var_rand, alpha=0.3, label="random ALR points")
    plt.scatter(d_train, var_train, color="red", s=30, label="training points")
    plt.xlabel("Distance to nearest training point (normalized space)")
    plt.ylabel("Predictive variance")
    plt.title("GP Predictive Variance vs. Distance to Training Data")
    plt.legend()
    plt.tight_layout()
    _finalize(fig, outpath=outpath, show=show)

    if points_csv is not None:
        points_csv = Path(points_csv)
        points_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "d_rand": d_rand.numpy(),
                "var_rand": var_rand.numpy(),
            }
        )
        df.to_csv(points_csv, index=False)


# ---------------------------------
# LOO metrics (+ within ±2σ count)
# ---------------------------------
def compute_and_save_loo_metrics(
    model,
    train_x: torch.Tensor,
    train_obj: torch.Tensor,
    metrics_csv: Path,
    metrics_json: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Compute LOO means/stds, RMSE, MAE, and % within ±2σ; save to CSV/JSON.
    """
    metrics_csv = Path(metrics_csv)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.likelihood.eval()
    loo_means, loo_stds = loo_predictive(model, train_x, train_obj)

    mse = torch.mean((loo_means - train_obj) ** 2).item()
    rmse = math.sqrt(mse)
    mae = torch.mean(torch.abs(loo_means - train_obj)).item()

    within = (
        ((train_obj >= loo_means - 2 * loo_stds) & (train_obj <= loo_means + 2 * loo_stds))
        .sum()
        .item()
    )
    total = int(train_obj.size(0))
    pct_within = 100.0 * within / total if total > 0 else float("nan")

    out = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "n_train": total,
        "d": int(train_x.shape[1]),
        "within_2sigma_count": within,
        "within_2sigma_percent": pct_within,
    }
    pd.DataFrame([out]).to_csv(metrics_csv, index=False)

    if metrics_json is not None:
        with open(metrics_json, "w") as f:
            json.dump(out, f, indent=2)

    return out


# ---------------------------------
# Convenience: run full diagnostics suite
# ---------------------------------
def run_full_diagnostics(
    model,
    train_x: torch.Tensor,
    train_obj: torch.Tensor,
    history: Dict[str, List[float]],
    figures_dir: Path,
    tables_dir: Path,
):
    """
    Generate + save all standard diagnostics (plots & CSVs).
    """
    figures_dir = Path(figures_dir)
    tables_dir = Path(tables_dir)
    (figures_dir / "gp").mkdir(parents=True, exist_ok=True)

    # 1) training curves
    plot_training_loss(history, outpath=figures_dir / "gp" / "loss_curve.pdf")
    plot_hyperparam_trajectories(history, outpath=figures_dir / "gp" / "hyperparam_trajectories.pdf")

    # 2) learned hyperparameters
    export_learned_hyperparameters(model, tables_dir / "gp_hyperparameters.csv")

    # 3) parity plot (+ R²)
    r2 = parity_plot(model, train_x, train_obj, outpath=figures_dir / "gp" / "parity_plot.pdf")

    # 4) LOO metrics (also save JSON for convenience)
    compute_and_save_loo_metrics(
        model,
        train_x,
        train_obj,
        metrics_csv=tables_dir / "gp_metrics.csv",
        metrics_json=tables_dir / "gp_metrics.json",
    )

    # 5) variance vs. distance (plus raw points CSV)
    variance_vs_distance_plot(
        model,
        train_x,
        outpath=figures_dir / "gp" / "variance_vs_distance.pdf",
        points_csv=tables_dir / "variance_vs_distance_points.csv",
    )

    # Return R² in case the caller wants it
    return {"r2": r2}