# src/lnp_bo/models/acquisition.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor

from botorch.acquisition import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler

from .bounds import alr_to_simplex, project_box_simplex


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def make_mc_sampler(n_mc: int) -> SobolQMCNormalSampler:
    """Sobol QMC normal sampler for MC acquisitions."""
    return SobolQMCNormalSampler(sample_shape=torch.Size([int(n_mc)]))


# ---------------------------------------------------------------------------
# Acquisition factories
# ---------------------------------------------------------------------------

def get_acquisition(
    name: str,
    *,
    model,
    train_x: Optional[Tensor] = None,
    train_y: Optional[Tensor] = None,
    sampler: Optional[SobolQMCNormalSampler] = None,
    objective: Optional[GenericMCObjective] = None,
    beta_ucb: float = 0.05,
):
    """
    Create a batch-compatible acquisition function.

    Parameters
    ----------
    name : one of {"LogNEI","LogEI","NEI","EI","PI","UCB"}
    model : fitted BoTorch model
    train_x, train_y : needed for NEI/PI/EI's best_f or baseline
    sampler : MC sampler (required for MC acquisitions)
    objective : GenericMCObjective (optional for single-output models)
    beta_ucb : exploration parameter for UCB
    """
    key = name.strip().upper()
    if key == "LOGNEI":
        if train_x is None:
            raise ValueError("LogNEI requires train_x as X_baseline.")
        return qLogNoisyExpectedImprovement(model=model, X_baseline=train_x, sampler=sampler, objective=objective)

    if key == "NEI":
        if train_x is None:
            raise ValueError("NEI requires train_x as X_baseline.")
        return qNoisyExpectedImprovement(model=model, X_baseline=train_x, sampler=sampler, objective=objective)

    # For EI / PI, use best_f from observed targets
    if key in {"LOGEI", "EI", "PI"}:
        if train_y is None:
            raise ValueError(f"{name} requires train_y to compute best_f.")
        best_f = float(train_y.max().item())

        if key == "LOGEI":
            return qLogExpectedImprovement(model=model, best_f=best_f, sampler=sampler, objective=objective)
        if key == "EI":
            return qExpectedImprovement(model=model, best_f=best_f, sampler=sampler, objective=objective)
        if key == "PI":
            return qProbabilityOfImprovement(model=model, best_f=best_f, sampler=sampler, objective=objective)

    if key == "UCB":
        # UCB is analytic; no sampler needed
        return qUpperConfidenceBound(model=model, beta=float(beta_ucb), objective=objective)

    raise ValueError(f"Unknown acquisition name: {name!r}")


# ---------------------------------------------------------------------------
# Optimize wrapper
# ---------------------------------------------------------------------------

def optimize_acqf_with_restarts(
    acq_function,
    bounds: Tensor,
    *,
    q: int,
    num_restarts: int = 10,
    raw_samples: int = 512,
    return_best_only: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Thin wrapper around BoTorch's optimize_acqf with common defaults.

    Parameters
    ----------
    acq_function : BoTorch acquisition instance
    bounds       : (2, d) tensor
    q            : batch size
    num_restarts : multistart restarts
    raw_samples  : random samples for initialization
    """
    candidates, acq_val = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=int(q),
        num_restarts=int(num_restarts),
        raw_samples=int(raw_samples),
        return_best_only=return_best_only,
    )
    return candidates.detach(), (acq_val.detach() if acq_val is not None else None)


# ---------------------------------------------------------------------------
# High-level proposal API
# ---------------------------------------------------------------------------

@dataclass
class ProposedBatch:
    """Container for a proposed batch of points."""
    z_alr_np: Tensor        # (q, K) = (q, (K-1)+1)  [ALR dims + N/P]
    mix_simplex: Tensor     # (q, K)  four-part mixture after projection
    np_column: Tensor       # (q, 1)  clamped N/P
    candidates_5d: Tensor   # (q, K+1) concatenated [mix, N/P]
    acq_value: Optional[Tensor] = None  # value returned by optimize_acqf (if any)


def propose_batch(
    *,
    model,
    train_x: Tensor,
    train_y: Tensor,
    bounds_alr_np: Tensor,         # (2, d_z)  [ALR dims + N/P]
    simplex_lb: Tensor,            # (K,)
    simplex_ub: Tensor,            # (K,)
    acq_name: str = "LogNEI",
    q: int = 3,
    num_restarts: int = 10,
    raw_samples: int = 512,
    mc_samples: int = 256,
    objective: Optional[GenericMCObjective] = None,
    beta_ucb: float = 0.05,
    seed: Optional[int] = None,
) -> ProposedBatch:
    """
    Optimize the acquisition in ALR+N/P, then map to a feasible [L,U]∩simplex×[npr_lb,npr_ub].

    Notes
    -----
    - Assumes the last column of Z is N/P; the first K-1 columns are ALR log-ratios
      (with the K-th lipid as the ALR denominator).
    - Uses `project_box_simplex` to satisfy both sum-to-one and [L,U] bounds.
    """
    device, dtype = train_x.device, train_x.dtype
    if seed is not None:
        torch.manual_seed(int(seed))

    # Build MC sampler if needed
    sampler = make_mc_sampler(mc_samples) if acq_name.upper() != "UCB" else None

    # Construct acquisition
    acq = get_acquisition(
        acq_name,
        model=model,
        train_x=train_x,
        train_y=train_y,
        sampler=sampler,
        objective=objective,
        beta_ucb=beta_ucb,
    )

    # Optimize in ALR+N/P space
    Z, acq_val = optimize_acqf_with_restarts(
        acq,
        bounds=bounds_alr_np.to(device=device, dtype=dtype),
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        return_best_only=True,
    )  # Z: (q, (K-1)+1)

    # Split ALR vs N/P
    K = simplex_lb.numel()
    alr_z = Z[:, : K - 1]
    npr_z = Z[:, K - 1 : K]  # (q,1)

    # Invert ALR -> 4-part simplex and project into [L,U]∩simplex
    mix = alr_to_simplex(alr_z)
    mix = project_box_simplex(mix, simplex_lb.to(device), simplex_ub.to(device))

    # Clamp N/P into its bounds
    npr_lb = bounds_alr_np[0, -1].item()
    npr_ub = bounds_alr_np[1, -1].item()
    npr = torch.clamp(npr_z, min=float(npr_lb), max=float(npr_ub))

    # Concatenate final 5D candidates
    candidates = torch.cat([mix, npr], dim=1)

    return ProposedBatch(
        z_alr_np=Z,
        mix_simplex=mix,
        np_column=npr,
        candidates_5d=candidates,
        acq_value=acq_val,
    )


# ---------------------------------------------------------------------------
# Batch-gain utility A(q) and ΔA(q)
# ---------------------------------------------------------------------------

def measure_batch_gains(
    model,
    train_x: Tensor,
    train_y: Tensor,
    bounds_alr_np: Tensor,
    *,
    q_list: Iterable[int] = (1, 2, 3, 4, 5),
    num_restarts: int = 5,
    raw_samples: int = 128,
    mc_samples: int = 256,
    acq_name: str = "LogEI",
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Compute acquisition values A(q) for multiple batch sizes, then ΔA(q) = A(q)-A(q-1).

    Notes
    -----
    - Uses a fixed acquisition type across all q (default LogEI).
    - Returns plain floats on CPU for easy plotting.
    """
    sampler = make_mc_sampler(mc_samples) if acq_name.upper() != "UCB" else None
    acq = get_acquisition(
        acq_name,
        model=model,
        train_x=train_x,
        train_y=train_y,
        sampler=sampler,
        objective=None,
    )

    A: Dict[int, float] = {}
    prev = 0.0
    dA: Dict[int, float] = {}

    for q in q_list:
        cand, val = optimize_acqf_with_restarts(
            acq,
            bounds=bounds_alr_np,
            q=int(q),
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            return_best_only=True,
        )
        a_q = float(val.item()) if val is not None else float("nan")
        A[int(q)] = a_q
        dA[int(q)] = a_q - prev
        prev = a_q

    return A, dA


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "make_mc_sampler",
    "get_acquisition",
    "optimize_acqf_with_restarts",
    "ProposedBatch",
    "propose_batch",
    "measure_batch_gains",
]