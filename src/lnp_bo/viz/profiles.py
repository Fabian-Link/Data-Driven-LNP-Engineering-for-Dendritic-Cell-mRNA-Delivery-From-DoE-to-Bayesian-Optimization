# src/lnp_bo/viz/profiles.py
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt


def _model_device_dtype(model) -> Tuple[torch.device, torch.dtype]:
    """Infer device/dtype from a fitted SingleTaskGP-like model."""
    X = model.train_inputs[0]
    return X.device, X.dtype


def _to_tensor(a, dtype: torch.dtype, device: torch.device) -> Tensor:
    if isinstance(a, Tensor):
        return a.to(device=device, dtype=dtype)
    return torch.as_tensor(a, dtype=dtype, device=device)


def _alr_from_raw_mix(X_raw: np.ndarray) -> np.ndarray:
    """ALR transform (last component as denominator) from raw 4-part compositions."""
    eps = 1e-12
    X_raw = np.clip(X_raw, eps, 1.0)
    return np.log(X_raw[:, :3] / X_raw[:, [3]])


# ---------------------------------------------------------------------------
# 1D profiles
# ---------------------------------------------------------------------------

def plot_1d_profiles(
    model,
    df,
    *,
    lipids: Sequence[str],
    np_col: str = "N/P",
    n_points: int = 100,
    lipid_grid: Tuple[float, float] = (1e-3, 0.9),
    show: bool = True,
    title_prefix: str = "GP Mean Profile",
) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
    """
    For each lipid, sweep its raw fraction in [lipid_grid], hold others at their mean (renormalize),
    append N/P fixed at its mean; also produce an N/P profile at mean composition.

    Returns
    -------
    figures : dict mapping keys:
        - f"{lipid}_profile" -> (fig, ax)
        - "np_profile" -> (fig, ax)
    """
    device, dtype = _model_device_dtype(model)

    # Means from data
    X_raw = df[list(lipids)].to_numpy(float)     # (n,4)
    comp_mean = X_raw.mean(axis=0)               # (4,)
    np_mean = float(df[np_col].mean())

    out: Dict[str, Tuple[plt.Figure, plt.Axes]] = {}

    # One profile per lipid
    for idx, lipid in enumerate(lipids):
        grid = np.linspace(lipid_grid[0], lipid_grid[1], int(n_points))
        Xg = np.tile(comp_mean, (grid.size, 1))
        Xg[:, idx] = grid
        Xg /= Xg.sum(axis=1, keepdims=True)

        Z = _alr_from_raw_mix(Xg)                                # (n,3)
        Zt = np.hstack([Z, np_mean * np.ones((grid.size, 1))])   # (n,4)

        Zt_t = _to_tensor(Zt, dtype, device)

        model.eval(); model.likelihood.eval()
        with torch.no_grad():
            post = model.posterior(Zt_t)
            mean_pred = post.mean.squeeze(-1).cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grid, mean_pred, lw=2)
        ax.set_xlabel(lipid)
        ax.set_ylabel("Predicted log-Fluorescence")
        ax.set_title(f"{title_prefix} for {lipid}")
        fig.tight_layout()
        if show:
            plt.show()

        out[f"{lipid}_profile"] = (fig, ax)

    # N/P profile at mean composition
    grid_np = np.linspace(float(df[np_col].min()), float(df[np_col].max()), int(n_points))
    Xg = np.tile(comp_mean, (grid_np.size, 1))
    Xg /= Xg.sum(axis=1, keepdims=True)
    Z = _alr_from_raw_mix(Xg)
    Zt_np = np.hstack([Z, grid_np.reshape(-1, 1)])
    Zt_t = _to_tensor(Zt_np, dtype, device)

    model.eval(); model.likelihood.eval()
    with torch.no_grad():
        post = model.posterior(Zt_t)
        mean_pred_np = post.mean.squeeze(-1).cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid_np, mean_pred_np, lw=2)
    ax.set_xlabel("N/P ratio")
    ax.set_ylabel("Predicted log-Fluorescence")
    ax.set_title(f"{title_prefix} for N/P (mean mixture)")
    fig.tight_layout()
    if show:
        plt.show()

    out["np_profile"] = (fig, ax)
    return out


# ---------------------------------------------------------------------------
# 2D pair heatmaps
# ---------------------------------------------------------------------------

def plot_pair_heatmap(
    model,
    df,
    *,
    lipids: Sequence[str],
    i: int,
    j: int,
    np_col: str = "N/P",
    n: int = 60,
    lipid_grid: Tuple[float, float] = (1e-3, 0.9),
    show: bool = True,
    title_prefix: str = "GP mean over",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Heatmap for lipid i vs lipid j (others fixed at their mean), N/P fixed at its mean.

    Returns
    -------
    (fig, ax)
    """
    device, dtype = _model_device_dtype(model)

    X_raw = df[list(lipids)].to_numpy(float)
    np_mean = float(df[np_col].mean())
    others = [k for k in range(4) if k not in (i, j)]
    fixed = X_raw[:, others].mean(axis=0)

    grid_i = np.linspace(lipid_grid[0], lipid_grid[1], int(n))
    grid_j = np.linspace(lipid_grid[0], lipid_grid[1], int(n))
    Gi, Gj = np.meshgrid(grid_i, grid_j)

    Xg = np.zeros((n * n, 4), dtype=float)
    Xg[:, i] = Gi.ravel()
    Xg[:, j] = Gj.ravel()
    # Fill others with fixed means
    for idx2, k in enumerate(others):
        Xg[:, k] = fixed[idx2]
    # Close the compositions
    Xg /= Xg.sum(axis=1, keepdims=True)

    Z = _alr_from_raw_mix(Xg)
    Z = np.hstack([Z, np_mean * np.ones((n * n, 1))])
    Zt = _to_tensor(Z, dtype, device)

    model.eval(); model.likelihood.eval()
    with torch.no_grad():
        post = model.posterior(Zt)
        mean_pred = post.mean.squeeze(-1).cpu().numpy()

    M = mean_pred.reshape(n, n)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        M,
        origin="lower",
        extent=(grid_i.min(), grid_i.max(), grid_j.min(), grid_j.max()),
        aspect="auto",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Predicted log-Fluorescence")
    ax.set_xlabel(lipids[i])
    ax.set_ylabel(lipids[j])
    ax.set_title(f"{title_prefix} {lipids[i]}/{lipids[j]}")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_2d_profiles_grid(
    model,
    df,
    *,
    lipids: Sequence[str],
    np_col: str = "N/P",
    n: int = 60,
    lipid_grid: Tuple[float, float] = (1e-3, 0.9),
    show_each: bool = True,
) -> Dict[Tuple[int, int], Tuple[plt.Figure, plt.Axes]]:
    """
    Generate heatmaps for all unique lipid pairs (6 plots for 4 lipids).

    Returns
    -------
    figs : dict mapping (i,j) -> (fig, ax)
    """
    figs: Dict[Tuple[int, int], Tuple[plt.Figure, plt.Axes]] = {}
    for i in range(4):
        for j in range(i + 1, 4):
            fig, ax = plot_pair_heatmap(
                model,
                df,
                lipids=lipids,
                i=i,
                j=j,
                np_col=np_col,
                n=n,
                lipid_grid=lipid_grid,
                show=show_each,
            )
            figs[(i, j)] = (fig, ax)
    return figs


# ---------------------------------------------------------------------------
# Profiles around a provided optimum
# ---------------------------------------------------------------------------

def plot_profiles_at_optimum(
    model,
    *,
    lipids: Sequence[str],
    best_x: np.ndarray,
    best_np: float,
    n_points: int = 100,
    sweep_np_factor: Tuple[float, float] = (0.2, 1.5),
    show: bool = True,
    title_prefix: str = "GP Mean Profile at GP Optimum",
) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
    """
    For each lipid, sweep its raw fraction while fixing the others to `best_x` (renormalized);
    also sweep N/P around `best_np` with the mixture fixed to `best_x`.

    Parameters
    ----------
    best_x : array-like (4,), mixture on the simplex (sums to 1)
    best_np : float, GP-optimal N/P

    Returns
    -------
    dict with keys:
        - f"{lipid}_profile_opt" -> (fig, ax)
        - "np_profile_opt" -> (fig, ax)
    """
    device, dtype = _model_device_dtype(model)
    best_x = np.asarray(best_x, dtype=float)
    best_x = best_x / best_x.sum()

    out: Dict[str, Tuple[plt.Figure, plt.Axes]] = {}

    # Lipid profiles
    for idx, lipid in enumerate(lipids):
        grid = np.linspace(1e-3, 0.9, int(n_points))
        Xg = np.tile(best_x, (grid.size, 1))
        Xg[:, idx] = grid
        Xg /= Xg.sum(axis=1, keepdims=True)

        Z = _alr_from_raw_mix(Xg)
        Zt = np.hstack([Z, best_np * np.ones((grid.size, 1))])
        Zt_t = _to_tensor(Zt, dtype, device)

        model.eval(); model.likelihood.eval()
        with torch.no_grad():
            post = model.posterior(Zt_t)
            mean_pred = post.mean.squeeze(-1).cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grid, mean_pred, lw=2)
        ax.set_xlabel(lipid)
        ax.set_ylabel("Predicted log-Fluorescence")
        ax.set_title(f"{title_prefix} for {lipid}")
        fig.tight_layout()
        if show:
            plt.show()
        out[f"{lipid}_profile_opt"] = (fig, ax)

    # N/P sweep at best_x
    lo = max(1e-6, sweep_np_factor[0] * float(best_np))
    hi = sweep_np_factor[1] * float(best_np)
    grid_np = np.linspace(lo, hi, int(n_points))

    Xg = np.tile(best_x, (grid_np.size, 1))
    Xg /= Xg.sum(axis=1, keepdims=True)
    Z = _alr_from_raw_mix(Xg)
    Zt_np = np.hstack([Z, grid_np.reshape(-1, 1)])
    Zt_t = _to_tensor(Zt_np, dtype, device)

    model.eval(); model.likelihood.eval()
    with torch.no_grad():
        post = model.posterior(Zt_t)
        mean_pred_np = post.mean.squeeze(-1).cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid_np, mean_pred_np, lw=2)
    ax.set_xlabel("N/P ratio")
    ax.set_ylabel("Predicted log-Fluorescence")
    ax.set_title(f"{title_prefix} for N/P")
    fig.tight_layout()
    if show:
        plt.show()
    out["np_profile_opt"] = (fig, ax)

    return out
