from typing import Optional

import torch
from botorch.acquisition import (
    qExpectedImprovement,
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler


# ------------------------------------------------------------------
# Define the Acquisition Objective.
# ------------------------------------------------------------------
def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Scalar objective used by MC acquisitions: maximize first output column."""
    return Z[..., 0]


# Reusable MC objective instance
objective = GenericMCObjective(objective=obj_callable)


# ------------------------------------------------------------------
# Acquisition Optimization
# ------------------------------------------------------------------
def optimize_acqf_function(
    acq_func,
    bounds: torch.Tensor,
    batch_size: int,
    num_restarts: int,
    raw_samples: int,
) -> torch.Tensor:
    """
    Optimize an acquisition function with BoTorch's `optimize_acqf`.

    Returns
    -------
    torch.Tensor
        Candidate points (shape: [q, d]) in the same space as `bounds`.
    """
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    return candidates.detach()


def get_acquisition_function(
    acq_name: str,
    model,
    train_x: torch.Tensor,
    train_obj: torch.Tensor,
    sampler: SobolQMCNormalSampler,
    objective: GenericMCObjective,
    bounds: Optional[torch.Tensor] = None,
):
    """
    Factory for common batch MC acquisition functions.
    """
    if acq_name == "LogEI":
        return qLogExpectedImprovement(
            model=model,
            best_f=-train_obj.min(),
            sampler=sampler,
            objective=objective,
        )
    elif acq_name == "LogNEI":
        return qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=train_x,
            sampler=sampler,
            objective=objective,
        )
    elif acq_name == "PI":
        return qProbabilityOfImprovement(
            model=model,
            best_f=-train_obj.min(),
            sampler=sampler,
            objective=objective,
        )
    elif acq_name == "UCB":
        return qUpperConfidenceBound(
            model=model,
            beta=0.05,
            objective=objective,
        )
    else:
        raise ValueError(f"Unknown acquisition function: {acq_name}")


def measure_batch_gains(
    model,
    train_x: torch.Tensor,
    train_obj: torch.Tensor,
    bounds: torch.Tensor,
    q_list=(1, 2, 3, 4, 5),
    num_restarts: int = 5,
    raw_samples: int = 128,
    mc_samples: int = 256,
):
    """
    Quick utility to compare acquisition value vs. batch size for LogEI.
    Returns dicts: gains[q] and marginal[q].
    """
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
    gains = {}
    for q in q_list:
        acq = qLogExpectedImprovement(
            model=model,
            best_f=train_obj.max().item(),
            sampler=sampler,
        )
        _, acq_val = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            return_best_only=True,
        )
        gains[q] = acq_val.item()

    marginal = {}
    prev = 0.0
    for q in q_list:
        marginal[q] = gains[q] - prev
        prev = gains[q]

    return gains, marginal