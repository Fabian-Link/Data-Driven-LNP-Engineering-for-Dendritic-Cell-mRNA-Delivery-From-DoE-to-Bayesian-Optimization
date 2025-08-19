# src/lnp_bo/io/load.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch import Tensor
except Exception:  # pragma: no cover
    torch = None
    Tensor = object  # type: ignore

# gpytorch is only needed for the noise prior helper
try:
    from gpytorch.priors import GammaPrior
except Exception:  # pragma: no cover
    GammaPrior = None  # type: ignore


# ---------------------------------------------------------------------------
# Data schemas
# ---------------------------------------------------------------------------

DEFAULT_LIPID_COLS: Tuple[str, str, str, str] = (
    "PEG-DMG-2000",
    "Ionizable Lipid",
    "DOPE",
    "Cholesterol",
)
DEFAULT_NP_COL: str = "N/P"
DEFAULT_RESP_COL: str = "Fluoreszenz [RLU]"


@dataclass
class Dataset:
    """Container returned by `prepare_dataset`."""
    df: pd.DataFrame               # cleaned dataframe
    X_aug: np.ndarray              # shape (N, 4) -> [ALR(3), N/P]
    y_log: np.ndarray              # shape (N,)
    lipids: Tuple[str, str, str, str]
    np_col: str
    resp_col: str


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _coerce_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Safely coerce requested columns to numeric (float), preserving others."""
    df2 = df.copy()
    for c in cols:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    return df2


def _alr_from_raw_mix(X_raw: np.ndarray) -> np.ndarray:
    """
    ALR transform (last component as denominator) for 4-part compositions.
    Input rows should be positive and (approximately) sum to 1; function is robust.
    """
    eps = 1e-12
    X_raw = np.clip(X_raw, eps, 1.0)
    X_raw = X_raw / (X_raw.sum(axis=1, keepdims=True) + eps)
    return np.log(X_raw[:, :3] / X_raw[:, [3]])


# ---------------------------------------------------------------------------
# Loading & cleaning
# ---------------------------------------------------------------------------

def load_and_clean_excel(
    path: str,
    *,
    lipid_cols: Sequence[str] = DEFAULT_LIPID_COLS,
    np_col: str = DEFAULT_NP_COL,
    response_col: str = DEFAULT_RESP_COL,
    rename: Optional[Mapping[str, str]] = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Load an Excel sheet, select required columns, coerce to numeric, optionally rename, drop NAs.

    Parameters
    ----------
    path : str
        Path to .xlsx (pandas uses openpyxl under the hood).
    lipid_cols : 4 column names for the mixture parts (order matters).
    np_col : name of the N/P column.
    response_col : name of the response (RLU) column.
    rename : optional mapping to rename columns before selection.
    dropna : if True, drop rows with any NA in the selected columns.

    Returns
    -------
    Cleaned DataFrame containing only the selected columns in this order:
      [lipid_cols..., np_col, response_col]
    """
    df = pd.read_excel(path, header=0)

    if rename:
        df = df.rename(columns=dict(rename))

    keep = list(lipid_cols) + [np_col, response_col]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}. Found: {list(df.columns)}")

    df = _coerce_numeric(df, keep)
    if dropna:
        df = df.dropna(subset=keep).reset_index(drop=True)

    # Optional: warn if mixtures don't sum ~ 1 (but don't mutate).
    mix_sum = df[list(lipid_cols)].sum(axis=1)
    off = np.abs(mix_sum.values - 1.0)
    if np.percentile(off, 95) > 1e-3:
        # gentle heads-up; caller can choose to renormalize upstream if needed
        print(
            "[warn] Many mixture rows deviate from sum=1 (95th pct > 1e-3). "
            "Consider renormalization if this isn't expected."
        )

    return df[keep]


# ---------------------------------------------------------------------------
# Preparation (features/targets)
# ---------------------------------------------------------------------------

def preprocess_df_to_arrays(
    df: pd.DataFrame,
    *,
    lipid_cols: Sequence[str] = DEFAULT_LIPID_COLS,
    np_col: str = DEFAULT_NP_COL,
    response_col: str = DEFAULT_RESP_COL,
    log_eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce feature matrix X_aug = [ALR(lipids), N/P] and target y_log from a cleaned DataFrame.
    """
    X_raw = df[list(lipid_cols)].to_numpy(float)
    Z_alr = _alr_from_raw_mix(X_raw)                           # (N,3)
    NP = df[np_col].to_numpy(float).reshape(-1, 1)             # (N,1)
    X_aug = np.hstack([Z_alr, NP])                             # (N,4)

    y = df[response_col].to_numpy(float)
    y_log = np.log(y + float(log_eps))
    return X_aug, y_log


def prepare_dataset(
    path: str,
    *,
    lipid_cols: Sequence[str] = DEFAULT_LIPID_COLS,
    np_col: str = DEFAULT_NP_COL,
    response_col: str = DEFAULT_RESP_COL,
    rename: Optional[Mapping[str, str]] = None,
    log_eps: float = 1e-8,
) -> Dataset:
    """
    One-shot convenience: load → clean → arrays.
    """
    df = load_and_clean_excel(
        path,
        lipid_cols=lipid_cols,
        np_col=np_col,
        response_col=response_col,
        rename=rename,
    )
    X_aug, y_log = preprocess_df_to_arrays(
        df,
        lipid_cols=lipid_cols,
        np_col=np_col,
        response_col=response_col,
        log_eps=log_eps,
    )
    return Dataset(
        df=df,
        X_aug=X_aug,
        y_log=y_log,
        lipids=tuple(lipid_cols),  # type: ignore[arg-type]
        np_col=np_col,
        resp_col=response_col,
    )


# ---------------------------------------------------------------------------
# Torch conversion (optional)
# ---------------------------------------------------------------------------

def to_torch_tensors(
    X_aug: np.ndarray,
    y_log: np.ndarray,
    *,
    device: str | "torch.device" = "cpu",
    dtype: Optional["torch.dtype"] = None,
) -> Tuple["Tensor", "Tensor"]:
    """
    Convert arrays to torch tensors suitable for GP training: (N,4) and (N,1).
    """
    if torch is None:
        raise ImportError("PyTorch is not available. Install torch to use `to_torch_tensors`.")
    if dtype is None:
        dtype = torch.double
    tx = torch.as_tensor(X_aug, dtype=dtype, device=device)
    ty = torch.as_tensor(y_log.reshape(-1, 1), dtype=dtype, device=device)
    return tx, ty


# ---------------------------------------------------------------------------
# Noise prior from triplicates
# ---------------------------------------------------------------------------

def compute_noise_prior_from_triplicates(
    path: str,
    *,
    name_col: str,
    response_col: str = DEFAULT_RESP_COL,
    epsilon: float = 1e-8,
    alpha: float = 3.0,
):
    """
    Estimate a Gamma prior for the GaussianLikelihood noise:

        1) log-transform the response: log(y + eps)
        2) compute within-group variance across replicates (grouped by `name_col`)
        3) set a GammaPrior over the *standardized* noise variance:
               sigma2_std = sigma2_emp / Var(log y)
           using concentration=alpha, rate=beta s.t. E[sigma2_std]=alpha/beta = sigma2_std

    Returns
    -------
    sigma2_emp : float    # empirical variance of log(RLU) within groups
    sigma2_std : float    # standardized variance (unitless)
    noise_prior : gpytorch.priors.GammaPrior  # to pass to GaussianLikelihood(noise_prior=...)
    """
    if GammaPrior is None:
        raise ImportError("gpytorch is required for `compute_noise_prior_from_triplicates`.")

    raw = pd.read_excel(path, usecols=[name_col, response_col])
    raw = raw.dropna(subset=[name_col, response_col]).reset_index(drop=True)
    raw["_logy"] = np.log(raw[response_col].to_numpy(float) + float(epsilon))

    # within-name sample variance of log(RLU)
    var_by_name = raw.groupby(name_col)["_logy"].var(ddof=1)
    sigma2_emp = float(var_by_name.mean())  # average within-group variance

    # standardize by global variance of log(RLU)
    sigma_y = float(raw["_logy"].std(ddof=0))
    sigma2_std = float(sigma2_emp / (sigma_y ** 2 + 1e-12))

    beta = float(alpha / (sigma2_std + 1e-12))
    noise_prior = GammaPrior(concentration=alpha, rate=beta)
    return sigma2_emp, sigma2_std, noise_prior
