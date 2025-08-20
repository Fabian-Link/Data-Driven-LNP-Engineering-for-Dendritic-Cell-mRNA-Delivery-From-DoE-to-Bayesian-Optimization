# scripts/propose_candidates.py

from pathlib import Path
import sys
import yaml
import json
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from botorch.sampling.normal import SobolQMCNormalSampler

# Make `src/` importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Local imports
from utils.load import load_and_clean, preprocess
from models.gp import initialize_model
from models.bounds import compute_alr_bounds_from_simplex_bounds, alr_to_simplex
from models.acquisition import (
    objective,
    get_acquisition_function,
    optimize_acqf_function,
)

# -----------------------------
# Config paths
# -----------------------------
CONFIG_PATHS  = ROOT / "configs" / "paths.yaml"
CONFIG_BOUNDS = ROOT / "configs" / "bounds.yaml"
CONFIG_ACQ    = ROOT / "configs" / "acquisition.yaml"

# -----------------------------
# Defaults / fallbacks
# -----------------------------
RESULTS_DIR     = ROOT / "results"
CAND_DIR        = RESULTS_DIR / "candidates"
MODELS_DIR      = RESULTS_DIR / "models"
FIG_DIR         = RESULTS_DIR / "figures"
TAB_DIR         = RESULTS_DIR / "tables"
DATA_RAW        = ROOT / "data" / "raw" / "LNP_Formulation_All_Data_Mean.xlsx"
DATA_PROCESSED  = ROOT / "data" / "processed" / "training_ALR_NP.csv"

# dtype/device/seed
dtype  = torch.double
device = torch.device("cpu")
SEED   = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------
# Load paths.yaml (optional)
# -----------------------------
if CONFIG_PATHS.exists():
    with open(CONFIG_PATHS, "r") as f:
        cfg_paths = yaml.safe_load(f) or {}
    r = cfg_paths.get("results") or {}
    RESULTS_DIR = ROOT / r.get("root_dir", "results")  # optional root folder
    CAND_DIR    = ROOT / r.get("candidates_dir", "results/candidates")
    MODELS_DIR  = ROOT / r.get("models_dir",    "results/models")
    FIG_DIR     = ROOT / r.get("figures_dir",   "results/figures")
    TAB_DIR     = ROOT / r.get("tables_dir",    "results/tables")
    d = cfg_paths.get("data") or {}
    DATA_RAW       = ROOT / d.get("train_excel", str(DATA_RAW))
    DATA_PROCESSED = ROOT / d.get("processed_csv", str(DATA_PROCESSED))

# Ensure output dir exists
CAND_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load bounds.yaml
# -----------------------------
with open(CONFIG_BOUNDS, "r") as f:
    cfg_bounds = yaml.safe_load(f)

labels    = cfg_bounds["mixture_labels"]                 # 4 names
L_bounds  = np.array(cfg_bounds["L_bounds"], dtype=float)
U_bounds  = np.array(cfg_bounds["U_bounds"], dtype=float)
np_min    = float(cfg_bounds.get("np_min", 1.0))
np_max    = float(cfg_bounds.get("np_max", 12.0))

# -----------------------------
# Load acquisition.yaml
# -----------------------------
with open(CONFIG_ACQ, "r") as f:
    cfg_acq = yaml.safe_load(f)

acq_name = cfg_acq["acquisition"]["name"]                # "LogNEI", "LogEI", "PI", "UCB"
beta     = float(cfg_acq["acquisition"].get("beta", 0.05))
opt_q        = int(cfg_acq["optimization"].get("q", 3))
num_restarts = int(cfg_acq["optimization"].get("num_restarts", 10))
raw_samples  = int(cfg_acq["optimization"].get("raw_samples", 512))
mc_samples   = int(cfg_acq["optimization"].get("mc_samples", 256))

# -----------------------------
# Rebuild training tensors
# -----------------------------
# Prefer processed CSV (written by fit_gp.py) for exact reproducibility.
if Path(DATA_PROCESSED).exists():
    df_proc = pd.read_csv(DATA_PROCESSED)
    X_aug = df_proc[["alr_PEG_over_Chol", "alr_IL_over_Chol", "alr_DOPE_over_Chol", "N/P"]].values
    y_log = df_proc["log_Fluoreszenz_RLU"].values
else:
    # Fallback: read raw and recreate ALR+N/P
    df_raw = load_and_clean(DATA_RAW)
    _, y_log_series = preprocess(df_raw)
    X4 = df_raw[labels].values  # order must match labels
    X_alr = np.log(X4[:, :3] / X4[:, [3]])         # Cholesterol as denominator
    NP_col = df_raw["N/P"].values.reshape(-1, 1)
    X_aug = np.hstack([X_alr, NP_col])
    y_log = y_log_series.values

train_x  = torch.tensor(X_aug, dtype=dtype, device=device)
train_obj= torch.tensor(y_log, dtype=dtype, device=device).unsqueeze(-1)

# -----------------------------
# Rebuild model & load weights
# -----------------------------
STATE_PATH = MODELS_DIR / "gp_state_dict.pt"
if not STATE_PATH.exists():
    raise FileNotFoundError(
        f"Could not find model weights at {STATE_PATH}. "
        "Run scripts/fit_gp.py first."
    )

mll, model = initialize_model(train_x, train_obj, state_dict=None, noise_prior=None)
model.load_state_dict(torch.load(STATE_PATH, map_location=device))
model.eval(); model.likelihood.eval()
print(f"[propose] Loaded GP state_dict from {STATE_PATH}")

# -----------------------------
# Build ALR+N/P bounds tensor
# -----------------------------
simplex_lb = torch.tensor(L_bounds, dtype=dtype, device=device)
simplex_ub = torch.tensor(U_bounds, dtype=dtype, device=device)

alr_lb, alr_ub = compute_alr_bounds_from_simplex_bounds(simplex_lb, simplex_ub)
npr_lb = torch.tensor(np_min, dtype=dtype, device=device)
npr_ub = torch.tensor(np_max, dtype=dtype, device=device)

bounds = torch.stack(
    [torch.cat([alr_lb, npr_lb.view(1,)]), torch.cat([alr_ub, npr_ub.view(1,)])],
    dim=0,
)  # shape (2, 4)

# -----------------------------
# Set up acquisition & optimize
# -----------------------------
sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
acqf = get_acquisition_function(
    acq_name=acq_name,
    model=model,
    train_x=train_x,
    train_obj=train_obj,
    sampler=sampler,
    objective=objective,
    bounds=bounds,
)

cand_z = optimize_acqf_function(
    acq_func=acqf,
    bounds=bounds,
    batch_size=opt_q,
    num_restarts=num_restarts,
    raw_samples=raw_samples,
)  # (q, 4): [alr1, alr2, alr3, N/P]

# -----------------------------
# Map to mixture + clamp
# -----------------------------
alr_z = cand_z[:, :3]
np_z  = cand_z[:, 3:].clone()

mix = alr_to_simplex(alr_z)  # (q,4)

# clamp to [L, U], then renormalize to sum=1
L_t = torch.tensor(L_bounds, dtype=dtype, device=device)
U_t = torch.tensor(U_bounds, dtype=dtype, device=device)
mix = torch.max(mix, L_t)
mix = torch.min(mix, U_t)
mix = mix / mix.sum(dim=1, keepdim=True)

# clamp N/P
np_z = torch.clamp(np_z, min=npr_lb, max=npr_ub)

# Concatenate final candidates in original 5D space
candidates = torch.cat([mix, np_z], dim=1)  # (q, 5)

# -----------------------------
# Score expected log fluorescence
# -----------------------------
with torch.no_grad():
    # Evaluate posterior at ALR+NP to be consistent with model input space
    post = model.posterior(cand_z)
    expected_log_f = post.mean.squeeze(-1).cpu().numpy()

# -----------------------------
# Save CSV to results/candidates
# -----------------------------
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_csv = CAND_DIR / f"candidates_{acq_name}_{ts}.csv"

cols = labels + ["N/P"] + ["expected_log_Fluoreszenz"]
arr  = np.hstack([candidates.cpu().numpy(), expected_log_f.reshape(-1, 1)])
pd.DataFrame(arr, columns=cols).to_csv(out_csv, index=False)

print(f"[propose] Wrote {opt_q} candidate(s) → {out_csv}")

# (Optional) also dump a tiny sidecar JSON with the settings used
sidecar = {
    "acquisition": acq_name,
    "beta": beta if acq_name == "UCB" else None,
    "q": opt_q,
    "num_restarts": num_restarts,
    "raw_samples": raw_samples,
    "mc_samples": mc_samples,
    "bounds_yaml": str(CONFIG_BOUNDS),
    "acq_yaml": str(CONFIG_ACQ),
    "paths_yaml": str(CONFIG_PATHS) if CONFIG_PATHS.exists() else None,
}
with open(out_csv.with_suffix(".json"), "w") as f:
    json.dump(sidecar, f, indent=2)
print("[propose] Sidecar JSON saved.")