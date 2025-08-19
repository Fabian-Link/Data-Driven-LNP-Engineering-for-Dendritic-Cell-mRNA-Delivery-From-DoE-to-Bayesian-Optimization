import pandas as pd
import numpy as np
from gpytorch.priors import GammaPrior

#==============================
# Load and process Excel data.
#==============================

# Data Loading and Cleaning
def load_and_clean(path) -> pd.DataFrame:
    """Read Excel file, promotes first row to header, coerce numerics and drops NaNs"""
    # Load in excel file as DataFrame
    df = pd.read_excel(path, header=0)
    # Selecting Columns
    keep = ["PEG-DMG-2000", "Ionizable Lipid", "DOPE", "Cholesterol", "N/P", "Fluoreszenz [RLU]"]
    # Force transform all entries to numeric
    df = df[keep].apply(pd.to_numeric, errors="coerce")
    # Drop rows that contain NaNs
    n_drop = df.isna().any(axis=1).sum()
    if n_drop:
        df = df.dropna().reset_index(drop=True)
    return df

# Preprocess function
def preprocess(df: pd.DataFrame):
    # Feature / Traget splot
    X = df[["PEG-DMG-2000", "Ionizable Lipid", "DOPE", "Cholesterol", "N/P"]]
    y = df["Fluoreszenz [RLU]"].values # Transform to np.ndarray for log transform
    y_log = np.log(y + 1e-8) # epsilon avoids log(0)
    return X, pd.Series(y_log, index=df.index)

def compute_noise_prior_from_triplicates(path, name_col, response_col, epsilon=1e-8, alpha=3):
    raw = pd.read_excel(path, usecols=[name_col, response_col])
    raw["_logy"] = np.log(raw[response_col] + epsilon)
    var_by_name = raw.groupby(name_col)["_logy"].var(ddof=1)
    sigma2_emp = var_by_name.mean()
    sigma_y = raw["_logy"].std(ddof=0)
    sigma2_std = sigma2_emp / (sigma_y ** 2)
    beta = alpha / sigma2_std
    noise_prior = GammaPrior(concentration=alpha, rate=beta)
    return sigma2_emp, sigma2_std, noise_prior