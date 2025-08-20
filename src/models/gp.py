import torch
import numpy as np
import random

from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior

from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms import Normalize


# -----------------------------
# Defaults (can be overridden by scripts)
# -----------------------------
dtype = torch.double
device = torch.device("cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ------------------------------------------------------------------
# GP Model Initialization
# ------------------------------------------------------------------
def initialize_model(
    train_x: torch.Tensor,
    train_obj: torch.Tensor,
    state_dict: dict | None = None,
    noise_prior: GammaPrior | None = None,
):
    """
    Build a SingleTaskGP with Normalize (inputs) and Standardize (outputs).
    Optionally inject a Gamma prior for the observation noise variance.

    Parameters
    ----------
    train_x : torch.Tensor (N, D)
        Training inputs in ALR+N/P space (already torch tensors).
    train_obj : torch.Tensor (N, 1)
        Training targets (log scale), as a column vector.
    state_dict : dict, optional
        If provided, load weights into the model.
    noise_prior : gpytorch.priors.GammaPrior, optional
        If given, used for GaussianLikelihood(noise_prior=...).

    Returns
    -------
    mll : ExactMarginalLogLikelihood
    model_obj : SingleTaskGP
    """
    base_kernel = RBFKernel(ard_num_dims=train_x.shape[-1])
    covar_module = ScaleKernel(base_kernel=base_kernel)

    if noise_prior is not None:
        likelihood = GaussianLikelihood(noise_prior=noise_prior)
    else:
        likelihood = GaussianLikelihood()

    model_obj = SingleTaskGP(
        train_X=train_x,
        train_Y=train_obj,
        covar_module=covar_module,
        likelihood=likelihood,
        input_transform=Normalize(d=train_x.shape[-1]),
        outcome_transform=Standardize(m=1),
    )

    if state_dict is not None:
        try:
            model_obj.load_state_dict(state_dict)
        except Exception as e:
            print("Warning: State dict loading failed, reinitializing model.", e)

    mll = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)
    return mll, model_obj


# ------------------------------------------------------------------
# Training callback helper (for logging hyperparameters)
# ------------------------------------------------------------------
def make_tracking_callback(model, history: dict):
    """
    Return a closure that logs loss and key hyperparameters during training.

    history should contain keys: 'loss', 'lengthscale', 'outputscale', 'noise'.
    """
    def _callback(params: dict[str, torch.Tensor], result):
        history["loss"].append(result.fval)
        try:
            ls = model.covar_module.base_kernel.lengthscale.mean().item()
            os = model.covar_module.outputscale.item()
            nv = model.likelihood.noise.item()
        except Exception:
            ls = os = nv = float("nan")
        history["lengthscale"].append(ls)
        history["outputscale"].append(os)
        history["noise"].append(nv)
    return _callback


# ------------------------------------------------------------------
# Leave-One-Out predictive mean/std (analytical)
# ------------------------------------------------------------------
def loo_predictive(model, train_x: torch.Tensor, train_obj: torch.Tensor):
    """
    Compute LOO mean and std for each training point under the current model.
    Returns tensors of shape (N, 1) for means and stds (in original log-y space).
    """
    model.eval()
    model.likelihood.eval()

    tf = model.outcome_transform
    y_mu = tf.means.item()
    y_std = tf.stdvs.item()

    # standardize y
    y_s = (train_obj - y_mu) / y_std

    # normalized X (via model transform)
    X = model.input_transform(train_x)

    # K + noise*I
    K = model.covar_module(X).evaluate()
    noise = model.likelihood.noise.item()
    n = K.size(-1)
    Ky = K + noise * torch.eye(n, dtype=K.dtype, device=K.device)

    L = torch.linalg.cholesky(Ky)
    alpha = torch.cholesky_solve(y_s, L)
    inv = torch.cholesky_inverse(L)

    inv_diag = inv.diag()
    loo_mean_std = y_s.squeeze() - alpha.squeeze() / inv_diag
    loo_var_std = 1.0 / inv_diag

    loo_mean = loo_mean_std * y_std + y_mu
    loo_std = torch.sqrt(loo_var_std) * y_std
    return loo_mean.unsqueeze(-1), loo_std.unsqueeze(-1)


# ------------------------------------------------------------------
# Examine Predictive Variances and Uncertainty
# ------------------------------------------------------------------
def distance_variance_profile(model, train_x: torch.Tensor, num_random: int = 1000):
    """
    Relate predictive variance to distance from training data (in normalized space).

    Returns
    -------
    d_train : (N,) zeros
    var_train : (N,) predictive variance at training points
    d_rand : (num_random,) distance of random points to nearest train point
    var_rand : (num_random,) predictive variance at random points
    """
    model.eval()
    model.likelihood.eval()

    Xn_train = model.input_transform(train_x)

    with torch.no_grad():
        var_train = model.posterior(train_x).variance.squeeze(-1).cpu()

    dim = train_x.shape[-1]
    Xr_raw = torch.randn(num_random, dim, dtype=train_x.dtype, device=train_x.device)
    Xn_rand = model.input_transform(Xr_raw)

    with torch.no_grad():
        var_rand = model.posterior(Xr_raw).variance.squeeze(-1).cpu()

    dmat = torch.cdist(Xn_rand.cpu(), Xn_train.cpu())
    d_rand = dmat.min(dim=1).values
    d_train = torch.zeros(train_x.size(0), dtype=d_rand.dtype)

    return d_train, var_train, d_rand, var_rand


# --- Back-compat alias so older scripts can import `tracking_callback` ---
def tracking_callback(model, history):
    return make_tracking_callback(model, history)