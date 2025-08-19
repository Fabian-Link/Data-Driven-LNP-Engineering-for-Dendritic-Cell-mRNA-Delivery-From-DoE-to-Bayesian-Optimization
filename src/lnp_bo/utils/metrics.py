# src/lnp_bo/utils/metrics.py
from __future__ import annotations

from typing import Iterable, Mapping, Optional, Tuple, Dict, Any
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# ----------------------------
# Helpers
# ----------------------------

def _to_numpy(x) -> np.ndarray:
    """Accepts numpy arrays, pandas Series, python lists, or torch tensors; returns 1D np.ndarray."""
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.reshape(-1)
    return x.astype(float)


def _safe_div(a: np.ndarray, b: np.ndarray | float, eps: float = 1e-12) -> np.ndarray:
    return a / (b + eps)


# ----------------------------
# Core regression metrics
# ----------------------------

def mse(y_true, y_pred) -> float:
    """Mean squared error."""
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_pred)
    return float(np.mean((y_t - y_p) ** 2))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_pred)
    return float(np.mean(np.abs(y_t - y_p)))


def r2(y_true, y_pred) -> float:
    """
    Coefficient of determination (R^2).
    Same definition as sklearn: 1 - SS_res / SS_tot (with SS_tot on centered y_true).
    """
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_pred)
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    # if y_true is constant, sklearn returns 0.0; we follow that behavior.
    if ss_tot <= 1e-18:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def mape(y_true, y_pred, eps: float = 1e-12) -> float:
    """Mean absolute percentage error (robust to zeros via eps)."""
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_pred)
    return float(np.mean(np.abs(_safe_div(y_t - y_p, np.maximum(np.abs(y_t), eps)))))


def smape(y_true, y_pred, eps: float = 1e-12) -> float:
    """Symmetric MAPE (bounded in [0, 2])."""
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_pred)
    denom = np.maximum(np.abs(y_t) + np.abs(y_p), eps)
    return float(2.0 * np.mean(np.abs(y_t - y_p) / denom))


# ----------------------------
# GP / probabilistic diagnostics
# ----------------------------

def residuals(y_true, y_pred) -> np.ndarray:
    """Return residuals y_true - y_pred as 1D numpy array."""
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_pred)
    return y_t - y_p


def standardized_residuals(y_true, mean, std, eps: float = 1e-12) -> np.ndarray:
    """
    Standardized residuals: (y_true - mean) / std.
    Useful for calibration checks; std should be predictive std (same units as y).
    """
    y_t = _to_numpy(y_true)
    mu = _to_numpy(mean)
    s = np.maximum(_to_numpy(std), eps)
    return (y_t - mu) / s


def coverage_within_k_sigma(
    y_true, mean, std, k: float = 2.0
) -> Tuple[float, int, int]:
    """
    Fraction of points with y_true ∈ [mean - k·std, mean + k·std].
    Returns (fraction, covered_count, total).
    """
    y_t = _to_numpy(y_true)
    mu = _to_numpy(mean)
    s = _to_numpy(std)
    covered = (y_t >= (mu - k * s)) & (y_t <= (mu + k * s))
    count = int(np.sum(covered))
    total = int(y_t.size)
    frac = float(count / total) if total else 0.0
    return frac, count, total


def gaussian_coverage_curve(
    y_true, mean, std, alphas: Iterable[float] = (0.5, 0.8, 0.9, 0.95, 0.99)
) -> Dict[float, float]:
    """
    For each alpha, compute empirical coverage for the central (100*alpha)% interval
    under a Gaussian assumption: [mu - z*std, mu + z*std] with z = Φ^{-1}((1+alpha)/2).
    """
    from scipy.stats import norm  # lazy import to keep base deps light

    y_t = _to_numpy(y_true)
    mu = _to_numpy(mean)
    s = _to_numpy(std)

    cover: Dict[float, float] = {}
    for a in alphas:
        z = float(norm.ppf(0.5 * (1.0 + float(a))))
        frac, _, _ = coverage_within_k_sigma(y_t, mu, s, k=z)
        cover[float(a)] = frac
    return cover


# ----------------------------
# One-shot summary
# ----------------------------

def regression_report(
    y_true,
    y_pred,
    *,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    cover_alphas: Iterable[float] = (0.95,),
) -> Mapping[str, Any]:
    """
    Bundle common regression metrics. If `mean` and `std` are provided (e.g. GP posterior),
    also include coverage and standardized residual stats.

    Returns a dict with floats (and small nested dicts for coverage).
    """
    report: Dict[str, Any] = {
        "n": int(_to_numpy(y_true).size),
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }

    if mean is not None and std is not None:
        # default: 95% empirical coverage and z-residual sanity checks
        cov = gaussian_coverage_curve(y_true, mean, std, alphas=cover_alphas)
        z = standardized_residuals(y_true, mean, std)
        report.update(
            {
                "coverage": cov,  # {alpha: fraction}
                "z_residual": {
                    "mean": float(np.mean(z)),
                    "std": float(np.std(z, ddof=1)) if z.size > 1 else 0.0,
                    "abs_mean": float(np.mean(np.abs(z))),
                    "max_abs": float(np.max(np.abs(z))) if z.size else 0.0,
                },
            }
        )

    return report


# ----------------------------
# __all__
# ----------------------------

__all__ = [
    "mse",
    "rmse",
    "mae",
    "r2",
    "mape",
    "smape",
    "residuals",
    "standardized_residuals",
    "coverage_within_k_sigma",
    "gaussian_coverage_curve",
    "regression_report",
]
