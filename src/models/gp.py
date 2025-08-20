import torch
from typing import Optional, Dict

from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim.core import OptimizationResult


def initialize_model(
    train_x: torch.Tensor,
    train_obj: torch.Tensor,
    noise_prior: Optional[object] = None,
    state_dict: Optional[dict] = None,
):
    """
    Build a SingleTaskGP with ARD RBF kernel, Gaussian likelihood, and standard
    Normalize/Standardize transforms. Optionally load a saved state_dict.
    Returns (mll, model).
    """
    base_kernel = RBFKernel(ard_num_dims=train_x.shape[-1])
    covar_module = ScaleKernel(base_kernel=base_kernel)
    likelihood = (
        GaussianLikelihood(noise_prior=noise_prior)
        if noise_prior is not None
        else GaussianLikelihood()
    )
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_obj,
        covar_module=covar_module,
        likelihood=likelihood,
        input_transform=Normalize(d=train_x.shape[-1]),
        outcome_transform=Standardize(m=1),
    )
    if state_dict is not None:
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            print("Warning: State dict loading failed, reinitializing model.", e)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def make_tracking_callback(model: SingleTaskGP, history: Dict[str, list]):
    """
    Return a fit callback that logs loss and key hyperparameters into `history`.
    Compatible with botorch.optim.fit.fit_gpytorch_mll_torch(..., callback=...).
    """
    def _cb(params: Dict[str, torch.Tensor], result: OptimizationResult):
        history.setdefault("loss", []).append(result.fval)
        history.setdefault("lengthscale", []).append(
            model.covar_module.base_kernel.lengthscale.mean().item()
        )
        history.setdefault("outputscale", []).append(
            model.covar_module.outputscale.item()
        )
        history.setdefault("noise", []).append(
            model.likelihood.noise.item()
        )
    return _cb


@torch.no_grad()
def loo_predictive(
    model: SingleTaskGP, train_x: torch.Tensor, train_obj: torch.Tensor
):
    """
    Fast leave-one-out (LOO) predictions using the trained model.
    Returns (loo_means, loo_stds), both shaped (N, 1) in the original (unstandardized) y space.
    """
    model.eval()
    model.likelihood.eval()

    # Outcome transform parameters
    tf = model.outcome_transform
    y_mu = tf.means.item()
    y_std = tf.stdvs.item()

    # Standardize targets
    y_s = (train_obj - y_mu) / y_std

    # Normalize inputs
    Xn = model.input_transform(train_x)

    # K + σ²I
    K = model.covar_module(Xn).evaluate()
    noise = model.likelihood.noise.item()
    n = K.size(-1)
    Ky = K + noise * torch.eye(n, dtype=K.dtype, device=K.device)

    # Cholesky-based solves
    L = torch.linalg.cholesky(Ky)
    alpha = torch.cholesky_solve(y_s, L)
    inv = torch.cholesky_inverse(L)

    inv_diag = inv.diag()
    loo_mean_std = y_s.squeeze() - alpha.squeeze() / inv_diag
    loo_var_std = 1.0 / inv_diag

    # Back-transform to original y space
    loo_mean = loo_mean_std * y_std + y_mu
    loo_std = torch.sqrt(loo_var_std) * y_std

    return loo_mean.unsqueeze(-1), loo_std.unsqueeze(-1)