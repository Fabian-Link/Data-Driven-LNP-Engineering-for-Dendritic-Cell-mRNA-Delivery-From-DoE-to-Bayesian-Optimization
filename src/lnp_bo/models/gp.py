import numpy as np
import random

import torch
from torch.optim import Adam

from gpytorch.kernels import RBFKernel
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms import Normalize
from botorch.optim.core import OptimizationResult
from botorch.optim.fit import fit_gpytorch_mll_torch

# -----------------------------
# Set device and dtype.
# -----------------------------
dtype = torch.double
device = torch.device("cpu")

# -----------------------------
# Set a global seed for reproducibility
# -----------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ------------------------------------------------------------------
# GP Model Initialization
# ------------------------------------------------------------------
def initialize_model(train_x, train_obj, state_dict=None):
    base_kernel = RBFKernel(
        #nu=1.5,
        ard_num_dims=train_x.shape[-1],
        #lengthscale=GammaPrior(alpha_ls, beta_ls)
    )
    covar_module = ScaleKernel(
        base_kernel=base_kernel,
        #outputscale_prior=GammaPrior(concentration=4, rate=5)
    )
    likelihood = GaussianLikelihood(noise_prior=noise_prior)
    model_obj = SingleTaskGP(
        train_X=train_x,
        train_Y=train_obj,
        covar_module=covar_module,
        likelihood=likelihood,
        input_transform=Normalize(d=train_x.shape[-1]),
        outcome_transform=Standardize(m=1)
    )
    d = train_x.size(-1)
    #model_obj.mean_module = LinearMean(d).to(train_x) 
    if state_dict is not None:
        try:
            model_obj.load_state_dict(state_dict)
        except Exception as e:
            print("Warning: State dict loading failed, reinitializing model.", e)
    mll = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)
    return mll, model_obj

def tracking_callback(params: dict[str, torch.Tensor],
                      result: OptimizationResult):
    # record the objective
    history["loss"].append(result.fval)
    # read the *constrained* values off the model
    #    (capture `model` by closure)
    ls = model.covar_module.base_kernel.lengthscale.mean().item()
    os = model.covar_module.outputscale.item()
    nv = model.likelihood.noise.item()
    history["lengthscale"].append(ls)
    history["outputscale"].append(os)
    history["noise"].append(nv)

# Initialize and fit the GP on the current (standardized) training data.
mll, model = initialize_model(train_x, train_obj)
model.train()
mll.train()

# ------------------------------------------------------------------
# GP Training
# ------------------------------------------------------------------
history = {
    "loss":        [],   # negative log-marginal-likelihood
    "lengthscale": [],
    "outputscale": [],
    "noise":       [],
}

# Fit with Adam via the torch‐based path, logging at each step
opt_res = fit_gpytorch_mll_torch(
    mll,
    optimizer=lambda params: Adam(params, lr=0.01),
    step_limit=800,
    callback=tracking_callback,
)

"""# Fit with L-BFGS-B, logging at each step
opt_res = fit_gpytorch_mll_scipy(
    mll,
    method="L-BFGS-B",
    options={"maxiter": 200},
    callback=tracking_callback,
)"""

# ------------------------------------------------------------------
# Leave-One-Out Cross-Validation
# ------------------------------------------------------------------

def loo_predictive(model, train_x, train_obj):
    # Get the model into eval modus
    model.eval()
    model.likelihood.eval()

    # Grab the Mean and Std of the GP outcome_transform
    tf = model.outcome_transform
    y_mu = tf.means.item()
    y_std = tf.stdvs.item()

    # Standardize the target
    y_s = (train_obj - y_mu) / y_std

    # Normalize the inputs via the models input_transform
    X = model.input_transform(train_x)

    # Build the full NxN covariance + noise matrix
    K = model.covar_module(X).evaluate() # Base covar
    noise = model.likelihood.noise.item() # Noise variance
    n = K.size(-1)
    Ky = K + noise * torch.eye(n, dtype=K.dtype, device=K.device)

    # Cholesky to invert the matrix
    L = torch.linalg.cholesky(Ky)
    alpha = torch.cholesky_solve(y_s, L)
    inv = torch.cholesky_inverse(L)

    # Invert the matrix
    inv_diag = inv.diag()
    loo_mean_std = y_s.squeeze() - alpha.squeeze() / inv_diag
    loo_var_std = 1.0 / inv_diag

    # Back transform from standardized space to the original y space (log-y)
    loo_mean = loo_mean_std * y_std + y_mu
    loo_std = torch.sqrt(loo_var_std) * y_std

    return loo_mean.unsqueeze(-1), loo_std.unsqueeze(-1)