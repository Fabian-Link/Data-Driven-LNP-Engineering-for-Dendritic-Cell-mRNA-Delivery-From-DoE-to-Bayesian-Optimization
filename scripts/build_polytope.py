# scripts/build_polytope.py
from pathlib import Path
import sys
import yaml
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

# Add <repo>/src to the import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from design.mixture_polytope import (
    compute_bounded_polytope_vertices,
    project_to_3d,
    compute_tetrahedron_volume,
)
from viz.mixture_plots import (
    get_common_axis_limits,
    plot_unconstrained_simplex,
    plot_polytope,
    plot_tetrahedral_subdivisions,
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
bounds_cfg = load_yaml(ROOT / "configs" / "bounds.yaml")
paths_cfg  = load_yaml(ROOT / "configs" / "paths.yaml")

labels = bounds_cfg["mixture_labels"]                     # e.g. ["PEG-DMG-2000","Ionizable Lipid","DOPE","Cholesterol"]
L_bounds = np.array(bounds_cfg["L_bounds"], dtype=float)  # length 4
U_bounds = np.array(bounds_cfg["U_bounds"], dtype=float)  # length 4
T_matrix = np.array(bounds_cfg["projection_matrix"], dtype=float)  # shape (4,3)

# Output directories
FIG_DIR   = ROOT / paths_cfg["results"]["figures_dir"]
DESIGN_DIR= ROOT / paths_cfg["results"]["design_dir"]
FIG_DIR.mkdir(parents=True, exist_ok=True)
DESIGN_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Build geometry
# -----------------------
# 4D vertices of bounded polytope
polytope_vertices_x = compute_bounded_polytope_vertices(L_bounds, U_bounds)
df_polytope_vertices = pd.DataFrame(polytope_vertices_x, columns=labels)

# 3D projection of vertices
proj_vertices = project_to_3d(polytope_vertices_x, T_matrix)
df_proj_vertices = pd.DataFrame(proj_vertices, columns=["X", "Y", "Z"])
common_limits = get_common_axis_limits(proj_vertices)

# Tetrahedralization in 3D + volumes
delaunay_obj = Delaunay(proj_vertices)
simplices = delaunay_obj.simplices  # (n_tetra, 4)
volumes = np.array([compute_tetrahedron_volume(proj_vertices[s]) for s in simplices])
volumes_norm = volumes / volumes.sum()
df_tetra = pd.DataFrame({
    "i0": simplices[:, 0],
    "i1": simplices[:, 1],
    "i2": simplices[:, 2],
    "i3": simplices[:, 3],
    "volume": volumes,
    "volume_normalized": volumes_norm,
})

# -----------------------
# Save artifacts
# -----------------------
(df_polytope_vertices
 ).to_csv(DESIGN_DIR / "polytope_vertices_x.csv", index=False)
(df_proj_vertices
 ).to_csv(DESIGN_DIR / "projected_vertices_3d.csv", index=False)
(df_tetra
 ).to_csv(DESIGN_DIR / "tetrahedra_info.csv", index=False)

# -----------------------
# Save plots (PDF, 300 dpi)
# -----------------------
plot_unconstrained_simplex(
    T_matrix,
    outpath=FIG_DIR / "simplex.pdf",
    dpi=300,
    show=False,
)
plot_polytope(
    proj_vertices,
    common_limits,
    outpath=FIG_DIR / "polytope.pdf",
    dpi=300,
    show=False,
)
plot_tetrahedral_subdivisions(
    proj_vertices,
    simplices,
    common_limits,
    outpath=FIG_DIR / "polytope_tetrahedra.pdf",
    dpi=300,
    show=False,
)

print("[build_polytope] Saved:")
print("  -", DESIGN_DIR / "polytope_vertices_x.csv")
print("  -", DESIGN_DIR / "projected_vertices_3d.csv")
print("  -", DESIGN_DIR / "tetrahedra_info.csv")
print("  -", FIG_DIR / "simplex.pdf")
print("  -", FIG_DIR / "polytope.pdf")
print("  -", FIG_DIR / "polytope_tetrahedra.pdf")