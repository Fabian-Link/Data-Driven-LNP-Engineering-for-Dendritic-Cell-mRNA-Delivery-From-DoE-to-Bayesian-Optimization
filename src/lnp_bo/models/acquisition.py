from typing import Optional

import torch

from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition import (
    qExpectedImprovement,
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)

# ------------------------------------------------------------------
# Define the Acquisition Objective.
# ------------------------------------------------------------------
def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
    # maximize f.
    return Z[..., 0]

objective = GenericMCObjective(objective=obj_callable)

def get_acquisition_function(acq_name, model, train_x, train_obj, sampler, objective, bounds=None):
    """
    Returns a batch-compatible acquisition function instance based on the policy type.
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

# ------------------------------------------------------------------
# Acquisition Optimization Settings.
# ------------------------------------------------------------------
def optimize_acqf_function(acq_func, bounds, batch_size, num_restarts, raw_samples):
    """
    Optimize the acquisition function using BoTorch's optimize_acqf.
    Returns candidate points in ALR space.
    """
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    return candidates.detach()

# ------------------------------------------------------------------
# Measure batch gains for different batch sizes
# ------------------------------------------------------------------
def measure_batch_gains(
    model,
    train_x,
    train_obj,
    bounds,
    q_list=[1,2,3,4,5],
    num_restarts=5,
    raw_samples=128,
    mc_samples=256,
):
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
    gains = {}
    for q in q_list:
        acq = qLogExpectedImprovement(          # ← swap here
            model=model,
            best_f=train_obj.max().item(),
            sampler=sampler,
        )
        candidates, acq_val = optimize_acqf(
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