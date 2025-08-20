from pathlib import Path
import sys
import yaml
import math
import random
import numpy as np
import torch
from torch.optim import Adam
import pandas as pd

# Make `src/` importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpytorch.priors import GammaPrior

from utils.load import load_and_clean, preprocess
from models.gp import (
    initialize_model,
    make_tracking_callback,
)
from viz.gp_diagnostics import run_full_diagnostics
from viz.profiles import plot_raw_profiles_all, plot_all_pairs, plot_gp_max_profiles
from models.exploitation_helper import exploit_gp_maximum

# -----------------------------
# Config
# -----------------------------
CONFIG_PATHS  = ROOT / "configs" / "paths.yaml"
CONFIG_MODEL  = ROOT / "configs" / "model.yaml"

# Defaults
RESULTS_FIG    = ROOT / "results" / "figures"
RESULTS_TAB    = ROOT / "results" / "tables"
RESULTS_MODELS = ROOT / "results" / "models"
DATA_FILE      = ROOT / "data" / "raw" / "LNP_Formulation_All_Data_Mean.xlsx"

TRAIN_STEPS = 800
TRAIN_LR    = 1e-2

# Load paths.yaml
if CONFIG_PATHS.exists():
    with open(CONFIG_PATHS, "r") as f:
        CFG_PATHS = yaml.safe_load(f) or {}
    rd = (CFG_PATHS.get("results") or {})
    RESULTS_FIG    = ROOT / rd.get("figures_dir", "results/figures")
    RESULTS_TAB    = ROOT / rd.get("tables_dir", "results/tables")
    RESULTS_MODELS = ROOT / rd.get("models_dir", "results/models")
    data_dict = CFG_PATHS.get("data") or {}
    DATA_FILE = ROOT / data_dict.get("train_excel", str(DATA_FILE))

# Ensure output dirs exist
RESULTS_FIG.mkdir(parents=True, exist_ok=True)
RESULTS_TAB.mkdir(parents=True, exist_ok=True)
RESULTS_MODELS.mkdir(parents=True, exist_ok=True)

# Load model.yaml (for noise prior + training params)
noise_prior = None
if CONFIG_MODEL.exists():
    with open(CONFIG_MODEL, "r") as f:
        CFG_MODEL = yaml.safe_load(f) or {}

    # Training overrides (optional)
    tr = CFG_MODEL.get("train") or {}
    TRAIN_STEPS = int(tr.get("steps", TRAIN_STEPS))
    TRAIN_LR    = float(tr.get("lr", TRAIN_LR))

    # Optional noise prior
    npcfg = CFG_MODEL.get("noise_prior") or {}
    use_prior = bool(npcfg.get("use", True))
    alpha = npcfg.get("alpha", None)
    beta  = npcfg.get("beta", None)
    if use_prior and (alpha is not None) and (beta is not None):
        noise_prior = GammaPrior(concentration=float(alpha), rate=float(beta))
        print(f"[fit_gp] Using Gamma noise prior: alpha={alpha}, beta={beta}")
    else:
        print("[fit_gp] No noise prior applied (default GaussianLikelihood).")

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
dtype = torch.double
device = torch.device("cpu")

# -----------------------------
# Load + preprocess data
# -----------------------------
df = load_and_clean(DATA_FILE)
X_df, y_log = preprocess(df)

# Build ALR+NP tensor as in the notebook:
X4 = df[["PEG-DMG-2000", "Ionizable Lipid", "DOPE", "Cholesterol"]].values
X_alr = np.log(X4[:, :3] / X4[:, [3]])          # ALR with Cholesterol as denominator
NP_col = df["N/P"].values.reshape(-1, 1)
X_aug = np.hstack([X_alr, NP_col])               # [alr1, alr2, alr3, N/P]

train_x = torch.tensor(X_aug, dtype=dtype, device=device)
train_obj = torch.tensor(y_log.values, dtype=dtype, device=device).unsqueeze(-1)

# -----------------------------
# NEW: Save processed dataset (what we feed to the GP)
# -----------------------------
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
processed_path = PROCESSED_DIR / "training_ALR_NP.csv"

df_processed = pd.DataFrame(
    np.hstack([X_aug, y_log.values.reshape(-1, 1)]),
    columns=[
        "alr_PEG_over_Chol",
        "alr_IL_over_Chol",
        "alr_DOPE_over_Chol",
        "N/P",
        "log_Fluoreszenz_RLU",
    ],
)
df_processed.to_csv(processed_path, index=False)
print(f"[fit_gp] Wrote processed training data → {processed_path}")

# -----------------------------
# Init & train GP
# -----------------------------
mll, model = initialize_model(train_x, train_obj, state_dict=None, noise_prior=noise_prior)
model.train()
mll.train()

history = {
    "loss": [],
    "lengthscale": [],
    "outputscale": [],
    "noise": [],
}
callback = make_tracking_callback(model, history)

from botorch.optim.fit import fit_gpytorch_mll_torch
opt_res = fit_gpytorch_mll_torch(
    mll,
    optimizer=lambda params: Adam(params, lr=TRAIN_LR),
    step_limit=TRAIN_STEPS,
    callback=callback,
)

# Save training history CSV
pd.DataFrame(history).to_csv(RESULTS_TAB / "gp_training_history.csv", index=False)

# -----------------------------
# Save model state
# -----------------------------
STATE_PATH = RESULTS_MODELS / "gp_state_dict.pt"
torch.save(model.state_dict(), STATE_PATH)
print(f"[fit_gp] Saved GP state_dict → {STATE_PATH}")

# -----------------------------
# Core diagnostics (plots + CSVs)
# -----------------------------
diag = run_full_diagnostics(
    model=model,
    train_x=train_x,
    train_obj=train_obj,
    history=history,
    figures_dir=RESULTS_FIG,
    tables_dir=RESULTS_TAB,
)
print(f"[fit_gp] Diagnostics complete. R²={diag['r2']:.3f}")

# -----------------------------
# Profiles @ dataset mean
# -----------------------------
lipids = ["PEG-DMG-2000", "Ionizable Lipid", "DOPE", "Cholesterol"]
profiles_dir = RESULTS_FIG / "profiles_mean"
pairs_dir    = RESULTS_FIG / "pairs_mean"

plot_raw_profiles_all(
    model=model,
    df=df,
    lipids=lipids,
    outdir=profiles_dir,
    prefix="profiles_mean",
    n_points=400,
)

plot_all_pairs(
    model=model,
    df=df,
    lipids=lipids,
    outdir=pairs_dir,
    prefix="pair_mean",
    n=70,
)

# -----------------------------
# Profiles @ GP optimum (heuristic exploitation)
# -----------------------------
best_x, best_np, best_mu = exploit_gp_maximum(
    model,
    dtype=train_x.dtype,
    device=train_x.device,
    n_samples=100000,   # decent coverage
    np_min=float(df["N/P"].min()),
    np_max=float(df["N/P"].max()),
    top_k=50,
    lbfgs_steps=100,
)
print("[fit_gp] Pure-exploitation optimum:")
for name, f in zip(lipids, best_x):
    print(f"  {name:15s}: {f:.3f}")
print(f"  N/P: {best_np:.3f}")
print(f"  Expected log-Fluorescence: {best_mu:.3f}")

profiles_opt_dir = RESULTS_FIG / "profiles_optimum"
plot_gp_max_profiles(
    model=model,
    lipids=lipids,
    best_x=best_x,
    best_np=best_np,
    outdir=profiles_opt_dir,
    prefix="gp_opt",
    n_points=400,
)

# Save pure-exploitation optimum to CSV
opt_row = {
    "PEG-DMG-2000": float(best_x[0]),
    "Ionizable Lipid": float(best_x[1]),
    "DOPE": float(best_x[2]),
    "Cholesterol": float(best_x[3]),
    "N/P": float(best_np),
    "Expected_log_Fluorescence": float(best_mu),
}
(RESULTS_TAB).mkdir(parents=True, exist_ok=True)
pd.DataFrame([opt_row]).to_csv(RESULTS_TAB / "gp_exploitation_optimum.csv", index=False)
print(f"[fit_gp] Saved pure-exploitation optimum → {RESULTS_TAB / 'gp_exploitation_optimum.csv'}")

print("[fit_gp] All plots + tables written.")
