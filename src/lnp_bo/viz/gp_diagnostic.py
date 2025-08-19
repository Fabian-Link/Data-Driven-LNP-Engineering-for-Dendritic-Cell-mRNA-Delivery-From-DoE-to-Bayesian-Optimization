# src/lnp_bo/viz/gp_diagnostics.py
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt

from lnp_bo.utils.metrics import r2 as r2_score_np
from lnp_bo.models.gp import (
    loo_predictive,
    random_variance_samples,
    distance_variance_profile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy_1d(x) -> np.ndarray:
    if isinstance(x, Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.reshape(-1)


def _posterior_mean_1d(model, X: Tensor) -> np.ndarray:
    model.eval(); model.likelihood.eval()
    with torch.no_grad():
        post = model.posterior(X)
        mu = post.mean.squeeze(-1).cpu().numpy()
    return mu


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_loss_curve(history: Dict[str, Iterable[float]], *, title: str = "Marginal Likelihood Optimization") -> Tuple[plt.Figure, plt.Axes]:
    """Plot negative log marginal likelihood vs iteration (from training history)."""
    iters = np.arange(len(history.get("loss", [])))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(iters, history.get("loss", []), label="Neg. log-lik")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_hyperparam_trajectories(history: Dict[str, Iterable[float]], *, title: str = "Hyperparameter Trajectories") -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """Plot ARD lengthscale (mean), outputscale, and noise variance over iterations."""
    iters = np.arange(len(history.get("loss", [])))

    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    axes[0].plot(iters, history.get("lengthscale", []), label="mean ARD ℓ")
    axes[0].legend()

    axes[1].plot(iters, history.get("outputscale", []), label="outputscale")
    axes[1].legend()

    axes[2].plot(iters, history.get("noise", []), label="noise var")
    axes[2].legend()
    axes[2].set_xlabel("Iteration")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, (axes[0], axes[1], axes[2])


# ---------------------------------------------------------------------------
# Parity / calibration
# ---------------------------------------------------------------------------

def plot_parity(
    model,
    X: Tensor,
    y: Tensor,
    *,
    title: str = "Parity Plot (train)",
    point_size: int = 50,
    alpha: float = 0.7,
) -> Tuple[plt.Figure, plt.Axes, float]:
    """
    Scatter of predicted vs true with 1:1 line. Returns (fig, ax, r2).
    """
    y_true = y.detach().cpu().numpy().reshape(-1)
    y_pred = _posterior_mean_1d(model, X)
    r2 = r2_score_np(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, s=point_size, alpha=alpha)
    mn, mx = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{title}\n$R^2$={r2:.3f}")
    fig.tight_layout()
    return fig, ax, r2


def loo_summary(
    model,
    train_x: Tensor,
    train_y: Tensor,
    *,
    k_sigma: float = 2.0,
) -> Dict[str, float]:
    """
    Compute LOO mean/std, RMSE/MAE, and ±kσ empirical coverage (default k=2).
    Returns a small dict (floats).
    """
    loo_mean, loo_std = loo_predictive(model, train_x, train_y)
    y = train_y.detach().cpu().numpy().reshape(-1)
    mu = loo_mean.detach().cpu().numpy().reshape(-1)
    sd = loo_std.detach().cpu().numpy().reshape(-1)

    rmse = float(np.sqrt(np.mean((mu - y) ** 2)))
    mae  = float(np.mean(np.abs(mu - y)))
    covered = np.logical_and(y >= (mu - k_sigma * sd), y <= (mu + k_sigma * sd))
    frac = float(np.mean(covered))
    return {
        "rmse": rmse,
        "mae": mae,
        f"coverage_pm_{k_sigma}sigma": frac,
    }


# ---------------------------------------------------------------------------
# Predictive variance diagnostics
# ---------------------------------------------------------------------------

def plot_predictive_variance_hist(
    model,
    *,
    num_samples: int = 2000,
    bins: int = 40,
    title: str = "Distribution of predictive variance over random ALR points",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw random ALR inputs and histogram the posterior variance (no training data involved).
    """
    rand_var = random_variance_samples(model, num_samples=num_samples)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rand_var.numpy(), bins=int(bins))
    ax.set_xlabel("Predictive variance")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_variance_vs_distance(
    model,
    train_x: Tensor,
    *,
    num_random: int = 1000,
    title: str = "GP Predictive Variance vs. Distance to Training Data",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Scatter of predictive variance vs distance to nearest training point in normalized space.
    Red markers highlight the training points (distance 0).
    """
    d_train, var_train, d_rand, var_rand = distance_variance_profile(model, train_x, num_random=num_random)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(d_rand.numpy(), var_rand.numpy(), alpha=0.3, label="random ALR points")
    ax.scatter(d_train.numpy(), var_train.numpy(), color="red", s=30, label="training points")
    ax.set_xlabel("Distance to nearest training point (normalized space)")
    ax.set_ylabel("Predictive variance")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax
