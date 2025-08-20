from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


# ---------------------------------
# Helpers
# ---------------------------------
def _finalize(fig, outpath: Optional[Path] = None, dpi: int = 300, show: bool = False):
    """Save/show/close a figure in a consistent way."""
    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight", transparent=True)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------
# 1D profiles at dataset means
# ---------------------------------
def plot_raw_profiles_all(
    model,
    df,
    lipids: Iterable[str],
    np_col: str = "N/P",
    n_points: int = 100,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    outdir: Optional[Path | str] = None,
    prefix: str = "profiles",
    show: bool = False,
) -> List[Path]:
    """
    Create 1D predictive profiles for each lipid (others at dataset mean & renormalized)
    and a profile over N/P (composition fixed at mean). Saves figures if `outdir` is provided.
    """
    paths: List[Path] = []

    X_raw = df[list(lipids)].values
    comp_mean = X_raw.mean(axis=0)
    np_mean = df[np_col].mean()

    # Lipid profiles
    for idx, lipid in enumerate(lipids):
        grid = np.linspace(1e-3, 0.9, n_points)
        Xg_raw = np.tile(comp_mean, (n_points, 1))
        Xg_raw[:, idx] = grid
        Xg_raw /= Xg_raw.sum(axis=1, keepdims=True)

        Z = np.log(Xg_raw[:, :-1] / Xg_raw[:, [-1]])
        Zt = np.hstack([Z, np_mean * np.ones((n_points, 1))])
        Zt_tensor = torch.tensor(Zt, dtype=dtype, device=device)

        model.eval(); model.likelihood.eval()
        with torch.no_grad():
            post = model.posterior(Zt_tensor)
            mean_pred = post.mean.squeeze(-1).cpu().numpy()

        fig = plt.figure(figsize=(6, 4))
        plt.plot(grid, mean_pred, lw=2)
        plt.xlabel(f"{lipid}")
        plt.ylabel("Predicted log-Fluorescence")
        plt.title(f"GP Mean Profile for {lipid}")
        plt.tight_layout()

        outpath = None
        if outdir is not None:
            outpath = Path(outdir) / f"{prefix}_lipid_{idx+1}_{lipid}.pdf"
            paths.append(outpath)
        _finalize(fig, outpath=outpath, show=show)

    # N/P profile (composition fixed at mean)
    grid_np = np.linspace(df[np_col].min(), df[np_col].max(), n_points)
    Xg_raw = np.tile(comp_mean, (n_points, 1))
    Xg_raw /= Xg_raw.sum(axis=1, keepdims=True)
    Z = np.log(Xg_raw[:, :-1] / Xg_raw[:, [-1]])
    Zt_np = np.hstack([Z, grid_np.reshape(-1, 1)])
    Zt_tensor = torch.tensor(Zt_np, dtype=dtype, device=device)

    model.eval(); model.likelihood.eval()
    with torch.no_grad():
        post = model.posterior(Zt_tensor)
        mean_pred_np = post.mean.squeeze(-1).cpu().numpy()

    fig = plt.figure(figsize=(6, 4))
    plt.plot(grid_np, mean_pred_np, lw=2)
    plt.xlabel("N/P ratio")
    plt.ylabel("Predicted log-Fluorescence")
    plt.title("GP Mean Profile for N/P")
    plt.tight_layout()

    outpath = None
    if outdir is not None:
        outpath = Path(outdir) / f"{prefix}_np.pdf"
        paths.append(outpath)
    _finalize(fig, outpath=outpath, show=show)

    return paths


# ---------------------------------
# 2D heatmaps / contours over lipid pairs
# ---------------------------------
def plot_pair(
    model,
    df,
    lipids: Iterable[str],
    i: int,
    j: int,
    n: int = 50,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    outpath: Optional[Path | str] = None,
    kind: str = "contour",  # "contour" or "heatmap"
    show: bool = False,
):
    """
    Sweep lipid i vs lipid j on an n×n grid (others at dataset means; renormalized),
    predict the GP mean, and plot as a contour map (default) or heatmap.
    """
    lipids = list(lipids)
    X_raw = df[lipids].values
    np_mean = df["N/P"].mean()

    others = [k for k in range(4) if k not in (i, j)]
    fixed = X_raw[:, others].mean(axis=0)

    grid_i = np.linspace(1e-3, 0.9, n)
    grid_j = np.linspace(1e-3, 0.9, n)
    Gi, Gj = np.meshgrid(grid_i, grid_j)

    Xg = np.zeros((n * n, 4))
    Xg[:, i] = Gi.ravel()
    Xg[:, j] = Gj.ravel()
    for idx2, k in enumerate(others):
        Xg[:, k] = fixed[idx2]
    Xg = Xg / Xg.sum(axis=1, keepdims=True)

    Z = np.log(Xg[:, :3] / Xg[:, [3]])
    Z = np.hstack([Z, np_mean * np.ones((n * n, 1))])
    Zt = torch.tensor(Z, dtype=dtype, device=device)

    model.eval(); model.likelihood.eval()
    with torch.no_grad():
        post = model.posterior(Zt)
        mean_pred = post.mean.squeeze(-1).cpu().numpy()

    M = mean_pred.reshape(n, n)

    fig = plt.figure(figsize=(5.5, 4.5))
    if kind == "contour":
        cs = plt.contourf(
            grid_i, grid_j, M, levels=30, antialiased=True
        )
        plt.colorbar(cs, label="Predicted log-Fluorescence")
    else:
        im = plt.imshow(
            M,
            origin="lower",
            extent=(grid_i.min(), grid_i.max(), grid_j.min(), grid_j.max()),
            aspect="auto",
        )
        plt.colorbar(im, label="Predicted log-Fluorescence")

    plt.xlabel(f"{lipids[i]}")
    plt.ylabel(f"{lipids[j]}")
    plt.title(f"GP mean over {lipids[i]} vs {lipids[j]}")
    plt.tight_layout()

    _finalize(fig, outpath=Path(outpath) if outpath else None, show=show)


def plot_all_pairs(
    model,
    df,
    lipids: Iterable[str],
    n: int = 60,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    outdir: Optional[Path | str] = None,
    prefix: str = "pair",
    kind: str = "contour",
    show: bool = False,
) -> List[Path]:
    """
    Generate all six lipid-pair maps (contour by default) and optionally save them.
    Returns list of saved file paths.
    """
    lipids = list(lipids)
    paths: List[Path] = []
    for i in range(4):
        for j in range(i + 1, 4):
            outpath = None
            if outdir is not None:
                outpath = Path(outdir) / f"{prefix}_{i+1}{j+1}_{lipids[i]}-{lipids[j]}.pdf"
                paths.append(outpath)
            plot_pair(
                model=model,
                df=df,
                lipids=lipids,
                i=i,
                j=j,
                n=n,
                dtype=dtype,
                device=device,
                outpath=outpath,
                kind=kind,
                show=show,
            )
    return paths


# ---------------------------------
# 1D profiles at GP optimum
# ---------------------------------
def plot_gp_max_profiles(
    model,
    lipids: Iterable[str],
    best_x: np.ndarray,
    best_np: float,
    n_points: int = 100,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    outdir: Optional[Path | str] = None,
    prefix: str = "gp_opt",
    show: bool = False,
) -> List[Path]:
    """
    At the GP-optimal mixture (best_x, best_np), sweep each lipid and N/P.
    """
    paths: List[Path] = []
    lipids = list(lipids)

    # Lipid profiles at the optimum mixture
    for idx, lipid in enumerate(lipids):
        grid = np.linspace(1e-3, 0.9, n_points)

        Xg = np.tile(best_x, (n_points, 1))
        Xg[:, idx] = grid
        Xg /= Xg.sum(axis=1, keepdims=True)

        Z = np.log(Xg[:, :-1] / Xg[:, [-1]])
        Zt = np.hstack([Z, best_np * np.ones((n_points, 1))])
        Zt_t = torch.tensor(Zt, dtype=dtype, device=device)

        model.eval(); model.likelihood.eval()
        with torch.no_grad():
            post = model.posterior(Zt_t)
            mean_pred = post.mean.squeeze(-1).cpu().numpy()

        fig = plt.figure(figsize=(6, 4))
        plt.plot(grid, mean_pred, lw=2)
        plt.xlabel(f"{lipid}")
        plt.ylabel("Predicted log-Fluorescence")
        plt.title(f"GP Mean Profile for {lipid} at GP Optimum")
        plt.tight_layout()

        outpath = None
        if outdir is not None:
            outpath = Path(outdir) / f"{prefix}_lipid_{idx+1}_{lipid}.pdf"
            paths.append(outpath)
        _finalize(fig, outpath=outpath, show=show)

    # N/P profile at the same mixture
    grid_np = np.linspace(best_np * 0.2, best_np * 1.5, n_points)
    Xg = np.tile(best_x, (n_points, 1))
    Xg /= Xg.sum(axis=1, keepdims=True)
    Z = np.log(Xg[:, :-1] / Xg[:, [-1]])
    Zt_np = np.hstack([Z, grid_np.reshape(-1, 1)])
    Zt_t = torch.tensor(Zt_np, dtype=dtype, device=device)

    model.eval(); model.likelihood.eval()
    with torch.no_grad():
        post = model.posterior(Zt_t)
        mean_pred_np = post.mean.squeeze(-1).cpu().numpy()

    fig = plt.figure(figsize=(6, 4))
    plt.plot(grid_np, mean_pred_np, lw=2)
    plt.xlabel("N/P ratio")
    plt.ylabel("Predicted log-Fluorescence")
    plt.title("GP Mean Profile for N/P at GP Optimum")
    plt.tight_layout()

    outpath = None
    if outdir is not None:
        outpath = Path(outdir) / f"{prefix}_np.pdf"
        paths.append(outpath)
    _finalize(fig, outpath=outpath, show=show)

    return paths