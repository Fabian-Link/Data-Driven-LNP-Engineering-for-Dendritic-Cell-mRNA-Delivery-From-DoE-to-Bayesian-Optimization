from typing import Tuple
import numpy as np
import torch


def compute_alr_bounds_from_simplex_bounds(
    lower: torch.Tensor,
    upper: torch.Tensor,
    num_samples: int = 20_000,
    batch: int = 5_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate ALR-space bounds implied by box constraints on a mixture simplex.

    Given per-component lower/upper bounds on a D-part composition x (with x>=0 and sum(x)=1),
    this function:
      1) rejection-samples compositions uniformly from Dirichlet(1) on Δ^{D-1},
      2) keeps those inside [lower, upper],
      3) maps them to ALR coordinates using the last part as the denominator,
      4) returns per-dimension min/max in ALR space.

    Parameters
    ----------
    lower : torch.Tensor, shape (D,)
        Lower bounds for each mixture component.
    upper : torch.Tensor, shape (D,)
        Upper bounds for each mixture component.
    num_samples : int, default 20000
        Total number of feasible samples to collect before computing min/max.
    batch : int, default 5000
        Batch size for each Dirichlet draw during rejection sampling.

    Returns
    -------
    alr_lb : torch.Tensor, shape (D-1,)
        Elementwise minimum in ALR space.
    alr_ub : torch.Tensor, shape (D-1,)
        Elementwise maximum in ALR space.

    Raises
    ------
    ValueError
        If bounds are inconsistent (e.g., lower>upper or infeasible simplex slice).

    Notes
    -----
    - Uses the (D-1)-dimensional ALR transform with the D-th component as reference.
    - Results are approximate and depend on `num_samples`.
    """
    if lower.shape != upper.shape:
        raise ValueError("lower and upper must have the same shape (D,).")
    if (lower > upper).any():
        raise ValueError("Each lower bound must be <= the corresponding upper bound.")
    if torch.sum(lower) > 1 + 1e-12 or torch.sum(upper) < 1 - 1e-12:
        raise ValueError("Infeasible simplex slice: ensure sum(lower) <= 1 <= sum(upper).")

    D = int(lower.numel())
    lower_np = lower.detach().cpu().numpy()
    upper_np = upper.detach().cpu().numpy()

    accepted = []
    max_loops = int(np.ceil(num_samples / max(1, batch)) * 50)  # generous cap to avoid infinite loops
    loops = 0

    while sum(a.shape[0] for a in accepted) < num_samples and loops < max_loops:
        Xs = np.random.dirichlet(np.ones(D), size=batch)  # (batch, D)
        mask = (Xs >= lower_np).all(axis=1) & (Xs <= upper_np).all(axis=1)
        if mask.any():
            accepted.append(Xs[mask])
        loops += 1

    if not accepted:
        raise ValueError("No feasible samples found; check your bounds.")

    Xall = np.vstack(accepted)[:num_samples]  # (N, D)
    Z = np.log(Xall[:, : D - 1] / Xall[:, [D - 1]])  # (N, D-1)

    alr_lb = torch.from_numpy(Z.min(axis=0)).to(lower)
    alr_ub = torch.from_numpy(Z.max(axis=0)).to(lower)
    return alr_lb, alr_ub


def alr_to_simplex(z: torch.Tensor) -> torch.Tensor:
    """
    Invert the Additive Log-Ratio (ALR) transform.

    Maps z ∈ R^{D-1} back to a D-part composition x on the simplex Δ^{D-1}
    using the last part as the reference component.

    Parameters
    ----------
    z : torch.Tensor, shape (N, D-1) or (D-1,)
        ALR coordinates.

    Returns
    -------
    x : torch.Tensor, shape (N, D) or (D,)
        Simplex point(s): nonnegative and summing to 1 along the last dimension.
    """
    if z.ndim == 1:
        z = z.view(1, -1)
        squeeze = True
    else:
        squeeze = False

    exp_z = torch.exp(z)                      # (N, D-1)
    ones = torch.ones(exp_z.size(0), 1, dtype=z.dtype, device=z.device)
    num = torch.cat([exp_z, ones], dim=1)     # (N, D)
    x = num / num.sum(dim=1, keepdim=True)    # normalize to simplex

    return x.squeeze(0) if squeeze else x