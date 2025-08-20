# scripts/design_generate.py
from pathlib import Path
import sys
import yaml
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

# Add <repo>/src to the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from design.sampling import (
    sobol_to_bary,
    generate_sobol_samples,
    generate_1d_sobol_samples,
    normalize_to_unit_cube,
    star_discrepancy,
)
from design.mixture_polytope import project_to_3d, compute_tetrahedron_volume
from viz.mixture_plots import (
    get_common_axis_limits,
    plot_polytope,
    plot_polytope_with_samples,
    plot_subdivisions_with_samples,
    plot_polytope_with_doe,
)

# -----------------------
# Load YAML helpers
# -----------------------
def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# -----------------------
# Config
# -----------------------
bounds_cfg  = load_yaml(ROOT / "configs" / "bounds.yaml")
design_cfg  = load_yaml(ROOT / "configs" / "design.yaml")
paths_cfg   = load_yaml(ROOT / "configs" / "paths.yaml")

labels      = bounds_cfg["mixture_labels"]  # only used for column names in 5D export
np_min      = float(bounds_cfg["np_min"])
np_max      = float(bounds_cfg["np_max"])

NUM_SOBOL_SAMPLES = int(design_cfg.get("num_sobol_samples", 16))
N_REPLICATES      = int(design_cfg.get("n_replicates", 100))
DOE_LIST          = design_cfg.get("doe", None)  # optional list of lists

FIG_DIR    = ROOT / paths_cfg["results"]["figures_dir"]
DESIGN_DIR = ROOT / paths_cfg["results"]["design_dir"]
FIG_DIR.mkdir(parents=True, exist_ok=True)
DESIGN_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Load geometry artifacts from build_polytope
# -----------------------
proj_vertices = pd.read_csv(DESIGN_DIR / "projected_vertices_3d.csv").to_numpy()
polytope_vertices_x = pd.read_csv(DESIGN_DIR / "polytope_vertices_x.csv").to_numpy()

common_limits = get_common_axis_limits(proj_vertices)

# Tetrahedra + volumes (recompute to keep the script standalone)
delaunay_obj = Delaunay(proj_vertices)
simplices = delaunay_obj.simplices  # (n_tetra, 4)
volumes = np.array([compute_tetrahedron_volume(proj_vertices[s]) for s in simplices])
volumes_norm = volumes / volumes.sum()

# -----------------------
# Random sample set (for comparison)
# -----------------------
rng = np.random.default_rng(42)
random_u = rng.random((NUM_SOBOL_SAMPLES, 3))
tet_idx_rand = rng.choice(len(simplices), size=NUM_SOBOL_SAMPLES, p=volumes_norm)
S3_rand, S4_rand = [], []
for i in range(NUM_SOBOL_SAMPLES):
    s = simplices[tet_idx_rand[i]]
    tet_proj = proj_vertices[s]
    tet_orig = polytope_vertices_x[s]
    bary = sobol_to_bary(random_u[i])
    S3_rand.append(bary @ tet_proj)
    S4_rand.append(bary @ tet_orig)
S3_rand = np.vstack(S3_rand)
S4_rand = np.vstack(S4_rand)
disc_rand = star_discrepancy(normalize_to_unit_cube(S3_rand), grid_points=30)

pd.DataFrame(S4_rand, columns=[f"x{i+1}" for i in range(polytope_vertices_x.shape[1])])\
  .to_csv(DESIGN_DIR / "random_sampled_points_4D.csv", index=False)

# -----------------------
# Sobol replicates → pick lowest discrepancy
# -----------------------
best_disc = np.inf
best_S3 = None
best_S4 = None

for _ in range(N_REPLICATES):
    sobol_pts = generate_sobol_samples(NUM_SOBOL_SAMPLES, d=3)
    tet_idx = rng.choice(len(simplices), size=NUM_SOBOL_SAMPLES, p=volumes_norm)
    S3, S4 = [], []
    for i in range(NUM_SOBOL_SAMPLES):
        s = simplices[tet_idx[i]]
        tet_proj = proj_vertices[s]
        tet_orig = polytope_vertices_x[s]
        bary = sobol_to_bary(sobol_pts[i])
        S3.append(bary @ tet_proj)
        S4.append(bary @ tet_orig)
    S3 = np.vstack(S3)
    S4 = np.vstack(S4)
    disc = star_discrepancy(normalize_to_unit_cube(S3), grid_points=30)
    if disc < best_disc:
        best_disc = disc
        best_S3 = S3
        best_S4 = S4

pd.DataFrame(best_S4, columns=[f"x{i+1}" for i in range(polytope_vertices_x.shape[1])])\
  .to_csv(DESIGN_DIR / "best_sobol_sampled_points_4D.csv", index=False)

# -----------------------
# Optional: DoE discrepancy
# -----------------------
disc_doe = None
if DOE_LIST:
    DOE_design = np.array(DOE_LIST, dtype=float)
    DOE_proj = project_to_3d(DOE_design, np.eye(4, 3))  # NOTE: DOE is already in 4D mixture; projection uses same T as build step
    # BUT we don't have T here. Project DOE using the same procedure as S4→S3:
    # use the same T that built proj_vertices: proj_vertices = X @ T
    # Recover T via least squares: T ≈ (X^T X)^{-1} X^T P
    X = polytope_vertices_x
    P = proj_vertices
    T_est, *_ = np.linalg.lstsq(X, P, rcond=None)
    DOE_proj = DOE_design @ T_est

    DOE_disc = star_discrepancy(normalize_to_unit_cube(DOE_proj), grid_points=30)
    disc_doe = float(DOE_disc)

    pd.DataFrame(DOE_design, columns=[f"x{i+1}" for i in range(polytope_vertices_x.shape[1])])\
      .to_csv(DESIGN_DIR / "DOE_design_4D.csv", index=False)
    pd.DataFrame(DOE_proj, columns=["X", "Y", "Z"])\
      .to_csv(DESIGN_DIR / "DOE_projected_3D.csv", index=False)

# -----------------------
# Add N/P via 1D Sobol and export 5D design (best Sobol)
# -----------------------
np_sobol = generate_1d_sobol_samples(NUM_SOBOL_SAMPLES, np_min, np_max).reshape(-1, 1)
best_S5 = np.hstack([best_S4, np_sobol])
cols_5d = labels + ["N/P"]  # mixture label names from YAML + N/P

pd.DataFrame(best_S5, columns=cols_5d)\
  .to_csv(DESIGN_DIR / "best_initial_5D_design.csv", index=False)

# -----------------------
# Discrepancy summary table
# -----------------------
rows = [
    {"design": "random_4D", "n": NUM_SOBOL_SAMPLES, "discrepancy": float(disc_rand)},
    {"design": "best_sobol_4D", "n": NUM_SOBOL_SAMPLES, "discrepancy": float(best_disc)},
]
if disc_doe is not None:
    rows.append({"design": "doe_4D", "n": len(DOE_LIST), "discrepancy": float(disc_doe)})

pd.DataFrame(rows).to_csv(DESIGN_DIR / "design_discrepancy_summary.csv", index=False)

# -----------------------
# Save plots (PDF, 300 dpi)
# -----------------------
plot_polytope(
    proj_vertices,
    common_limits,
    outpath=FIG_DIR / "polytope.pdf",      # overwrite/ensure present
    dpi=300,
    show=False,
)
plot_polytope_with_samples(
    proj_vertices,
    best_S3,
    "Polytope with Sobol Samples",
    common_limits,
    outpath=FIG_DIR / "polytope_sobol_samples.pdf",
    dpi=300,
    show=False,
)
plot_subdivisions_with_samples(
    proj_vertices,
    simplices,
    best_S3,
    common_limits,
    outpath=FIG_DIR / "polytope_subdivisions_sobol_samples.pdf",
    dpi=300,
    show=False,
)
plot_polytope_with_samples(
    proj_vertices,
    S3_rand,
    "Polytope with Random Samples",
    common_limits,
    outpath=FIG_DIR / "polytope_random_samples.pdf",
    dpi=300,
    show=False,
)
if DOE_LIST:
    plot_polytope_with_doe(
        proj_vertices,
        DOE_proj,
        common_limits,
        outpath=FIG_DIR / "polytope_doe.pdf",
        dpi=300,
        show=False,
    )

print("[design_generate] Saved:")
print("  -", DESIGN_DIR / "random_sampled_points_4D.csv")
print("  -", DESIGN_DIR / "best_sobol_sampled_points_4D.csv")
print("  -", DESIGN_DIR / "best_initial_5D_design.csv")
print("  -", DESIGN_DIR / "design_discrepancy_summary.csv")
if DOE_LIST:
    print("  -", DESIGN_DIR / "DOE_design_4D.csv")
    print("  -", DESIGN_DIR / "DOE_projected_3D.csv")
print("  -", FIG_DIR / "polytope_sobol_samples.pdf")
print("  -", FIG_DIR / "polytope_subdivisions_sobol_samples.pdf")
print("  -", FIG_DIR / "polytope_random_samples.pdf")
if DOE_LIST:
    print("  -", FIG_DIR / "polytope_doe.pdf")