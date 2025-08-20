import pandas as pd
import numpy as np


# -----------------------------
# Load and process Excel data.
# -----------------------------
def load_and_clean(path) -> pd.DataFrame:
    """
    Read Excel file, coerce numerics, keep required columns, drop rows with NaNs.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the Excel file.

    Returns
    -------
    df : pd.DataFrame
        Cleaned dataframe with required columns only.
    """
    df = pd.read_excel(path, header=0)
    keep = ["PEG-DMG-2000", "Ionizable Lipid", "DOPE", "Cholesterol", "N/P", "Fluoreszenz [RLU]"]
    df = df[keep].apply(pd.to_numeric, errors="coerce")
    n_drop = df.isna().any(axis=1).sum()
    if n_drop:
        df = df.dropna().reset_index(drop=True)
    return df


def preprocess(df: pd.DataFrame):
    """
    Split features/target and log-transform the response.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with columns: PEG-DMG-2000, Ionizable Lipid, DOPE, Cholesterol, N/P
    y_log : pd.Series
        Log-transformed response (log(RLU + 1e-8)).
    """
    X = df[["PEG-DMG-2000", "Ionizable Lipid", "DOPE", "Cholesterol", "N/P"]]
    y = df["Fluoreszenz [RLU]"].values
    y_log = np.log(y + 1e-8)  # epsilon avoids log(0)
    return X, pd.Series(y_log, index=df.index)
