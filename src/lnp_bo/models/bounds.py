# src/lnp_bo/models/bounds.py
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# ALR <-> simplex transforms
# ---------------------------------------------------------------------------

def simplex_to_alr(x: Tensor) -> Tensor:
    """
    Convert K-part compositions x (sum=1, x>0) to (K-1)-dim ALR coordinates,
    using the LAST component as the denominator.

    Parameters
    ----------
    x : (N, K) tensor

    Returns
    -------
    z : (N, K-1) tensor,  z_j = log(x_j / x_K)
    """
    if x.dim() != 2:
        raise ValueError("simplex_to_alr expects a (N, K) tensor.")
    # small epsilon for numerical safety (avoid log(0))
    eps = torch.finfo(x.dtype).eps
    num = torch.clamp(x[:, :-1], min=eps)
    den = torch.clamp(x[:, [-1]], min=eps)
    return torch.log(num / den)


def alr_to_simplex(z: Tensor) -> Tensor:
    """
    Invert ALR: (K-1)-dim => K-part composition on the simplex.

    Parameters
    ----------
    z : (N, K-1) tensor

    Returns
    -------
    x : (N, K) tensor with rows summing to 1
    """
    if z.dim() != 2:
        raise ValueError("alr_to_simplex expects a (N, K-1) tensor.")
    exp_z = torch.exp(z)
    ones = torch.ones(z.size(0), 1, dtype=z.dtype, device=z.device)
    num = torch.cat([exp_z, ones], dim=1)
    return num / num.sum(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Box-constrained simplex projection
#   Project y onto {x : sum(x)=1, L <= x <= U}
#   Solve x = clamp(y - tau, L, U) with sum(x)=1 for tau via bisection.
# ---------------------------------------------------------------------------

def _project_box_simplex_row(
    y: Tensor, L: Tensor, U: Tensor, *, tol: float = 1e-10, max_iter: int = 100
) -> Tensor:
    """
    Project a single K-vector y onto the intersection of the simplex and box:
      S = { x : sum(x) = 1,  L_i <= x_i <= U_i }.

    Returns the unique solution x = clamp(y - tau, L, U) with sum(x)=1.

    Preconditions: sum(L) <= 1 <= sum(U), and L <= U elementwise.
    """
    if not (torch.all(L <= U)):
        raise ValueError("Lower bounds must be <= upper bounds elementwise.")
    sL = float(L.sum().item())
    sU = float(U.sum().item())
    if sL > 1 + 1e-12 or sU < 1 - 1e-12:
        raise ValueError("Infeasible bounds: require sum(L) <= 1 <= sum(U).")

    # Monotone function f(tau) = sum(clamp(y - tau, L, U)) - 1
    # We search tau s.t. f(tau)=0 via bisection.
    # Bracket: when tau -> -inf, x -> U (sum -> sU);
    #          when tau -> +inf, x -> L (sum -> sL)
    # Find finite bracket by evaluating at tau candidates.
    def f(tau: float) -> float:
        x = torch.clamp(y - tau, min=L, max=U)
        return float(x.sum().item() - 1.0)

    # Start with wide bracket; expand if needed
    low, high = -1.0, 1.0
    f_low, f_high = f(low), f(high)
    expand = 0
    while f_low * f_high > 0 and expand < 50:
        # expand exponentially until signs differ
        low *= 2.0
        high *= 2.0
        f_low, f_high = f(low), f(high)
        expand += 1
    if f_low * f_high > 0:
        # As a fallback, clamp to closest bound and renormalize with a small repair
        x = torch.clamp(y, min=L, max=U)
        # If sum != 1 after clamp, shift mass within free coords by proportional scaling
        s = x.sum()
        if abs(float(s.item()) - 1.0) < tol:
            return x
        # distribute delta to variables not at bounds
        delta = 1.0 - float(s.item())
        free = (x > L + 1e-15) & (x < U - 1e-15)
        if free.any():
            share = delta / float(free.sum().item())
            x[free] = torch.clamp(x[free] + share, min=L[free], max=U[free])
            # One more normalization round by recursion
            return _project_box_simplex_row(x, L, U, tol=tol, max_iter=max_iter)
        # If all at bounds, return best-effort (shouldn't happen for feasible box)
        return x

    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        fm = f(mid)
        if abs(fm) <= tol:
            tau = mid
            break
        if f_low * fm < 0:
            high, f_high = mid, fm
        else:
            low, f_low = mid, fm
    else:
        tau = 0.5 * (low + high)

    x = torch.clamp(y - tau, min=L, max=U)
    # tiny numerical fix to ensure exact sum 1 within tolerance
    diff = float(x.sum().item()) - 1.0
    if abs(diff) > 1e-9:
        # distribute error to the largest slack variable to maintain bounds
        # (rarely needed)
        idx = torch.argmax(U - L).item()
        x[idx] = torch.clamp(x[idx] - diff, min=L[idx], max=U[idx])
    return x


def project_box_simplex(
    Y: Tensor, L: Tensor, U: Tensor, *, tol: float = 1e-10, max_iter: int = 100
) -> Tensor:
    """
    Batch version of the box-constrained simplex projection.

    Parameters
    ----------
    Y : (N, K) tensor
    L : (K,) tensor
    U : (K,) tensor

    Returns
    -------
    X : (N, K) tensor where each row lies in {sum=1, L<=x<=U}.
    """
    if Y.dim() != 2:
        raise ValueError("Y must be (N, K).")
    if L.dim() != 1 or U.dim() != 1 or L.numel() != Y.size(1) or U.numel() != Y.size(1):
        raise ValueError("L and U must be 1D tensors with length K == Y.shape[1].")

    X = torch.empty_like(Y)
    for i in range(Y.size(0)):
        X[i] = _project_box_simplex_row(Y[i], L, U, tol=tol, max_iter=max_iter)
    return X


# ---------------------------------------------------------------------------
# ALR bounds from simplex bounds (rejection sampling)
#   Matches your notebook logic, but generalized to K parts.
# ---------------------------------------------------------------------------

def compute_alr_bounds_from_simplex_bounds(
    lower: Tensor,
    upper: Tensor,
    *,
    num_samples: int = 20_000,
    batch: int = 5_000,
    seed: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Sample feasible compositions x ∈ Δ^{K-1} ∩ [lower, upper], convert to ALR
    (last part as denominator), and return per-dimension min/max.

    Parameters
    ----------
    lower, upper : (K,) tensors with sum(lower) <= 1 <= sum(upper)
    num_samples : total number of accepted samples to gather
    batch : proposal batch size for each Dirichlet draw
    seed : numpy RNG seed for reproducibility (sampling in numpy for speed)

    Returns
    -------
    (alr_lb, alr_ub) : two (K-1,) tensors on same device/dtype as `lower`.
    """
    if lower.dim() != 1 or upper.dim() != 1 or lower.numel() != upper.numel():
        raise ValueError("lower and upper must be 1D tensors of the same length K.")
    if torch.any(lower > upper):
        raise ValueError("Elementwise lower > upper found.")
    if float(lower.sum().item()) > 1.0 or float(upper.sum().item()) < 1.0:
        raise ValueError("Infeasible bounds: need sum(lower) <= 1 <= sum(upper).")

    K = lower.numel()
    rng = np.random.default_rng(seed)
    accepted = []

    lower_np = lower.detach().cpu().numpy()
    upper_np = upper.detach().cpu().numpy()

    while sum(a.shape[0] for a in accepted) < num_samples:
        # Dirichlet(1) proposals on the simplex
        Xs = rng.dirichlet(np.ones(K), size=batch)
        mask = (Xs >= lower_np - 1e-12).all(axis=1) & (Xs <= upper_np + 1e-12).all(axis=1)
        if np.any(mask):
            accepted.append(Xs[mask])

    Xall = np.vstack(accepted)[:num_samples]  # (num_samples, K)
    Z = np.log(Xall[:, : K - 1] / Xall[:, [K - 1]])  # ALR w/ last part as denom

    alr_lb = torch.as_tensor(Z.min(axis=0), dtype=lower.dtype, device=lower.device)
    alr_ub = torch.as_tensor(Z.max(axis=0), dtype=lower.dtype, device=lower.device)
    return alr_lb, alr_ub


# ---------------------------------------------------------------------------
# Convenience checks
# ---------------------------------------------------------------------------

def is_within_box_simplex(X: Tensor, L: Tensor, U: Tensor, tol: float = 1e-9) -> Tensor:
    """
    Per-row feasibility: L <= x <= U and |sum(x)-1|<=tol.

    Parameters
    ----------
    X : (N, K)
    L, U : (K,)

    Returns
    -------
    mask : (N,) boolean tensor
    """
    within = (X >= (L - tol)).all(dim=1) & (X <= (U + tol)).all(dim=1)
    sums_ok = (X.sum(dim=1) - 1.0).abs() <= tol
    return within & sums_ok


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "simplex_to_alr",
    "alr_to_simplex",
    "project_box_simplex",
    "compute_alr_bounds_from_simplex_bounds",
    "is_within_box_simplex",
]