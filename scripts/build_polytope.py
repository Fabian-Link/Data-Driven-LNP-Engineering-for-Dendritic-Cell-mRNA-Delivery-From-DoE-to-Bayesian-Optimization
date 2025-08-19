# scripts/build_polytope.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from lnp_bo.io.config import load_bounds, load_paths
from lnp_bo.design.mixture_polytope import (
    build_projected_triangulated_polytope,
    default_projection_matrix,
)
from lnp_bo.viz.mixture_plots import (
    get_common_axis_limits,
    plot_unconstrained_simplex,
    plot_polytope,
    plot_tetrahedral_subdivisions,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the bounded mixture polytope (vertices → 3D projection → Delaunay tetrahedra)."
    )
    p.add_argument("--bounds", default="configs/bounds.yaml", help="Path to bounds.yaml")
    p.add_argument("--paths", default="configs/paths.yaml", help="Path to paths.yaml")
    p.add_argument("--save-csv", action="store_true", help="Save vertices/triangulation tables to results/tables")
    p.add_argument("--save-figs", action="store_true", help="Save figures to results/figures")
    p.add_argument("--no-show", action="store_true", help="Do not show figures (useful for CI)")
    p.add_argument("--fig-suffix", default="", help="Optional suffix for figure filenames, e.g. _v1")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Load configs
    bounds = load_bounds(args.bounds)
    paths = load_paths(args.paths)

    L = np.asarray(bounds.lower, dtype=float)
    U = np.asarray(bounds.upper, dtype=float)
    d = L.size
    if d != 4:
        print(f"[error] This script assumes a 4-part mixture (got d={d}). Provide your own T in a custom script.", file=sys.stderr)
        return 2

    # Build vertices (x-space), project to 3D, and triangulate
    vertices_x, proj_vertices, tri = build_projected_triangulated_polytope(L, U)

    # Summary
    total_vol = float(np.sum(tri.volumes))
    print("=== Polytope summary ===")
    print(f"Dimensions (d):        {d}")
    print(f"# vertices:            {vertices_x.shape[0]}")
    print(f"# tetrahedra:          {tri.simplices.shape[0]}")
    print(f"Total 3D volume:       {total_vol:.6g}")
    print(f"Weights sum (check):   {float(tri.volumes_norm.sum()):.6f}")

    # Common axis limits for consistent viz
    limits = get_common_axis_limits(proj_vertices)

    # Plot (optionally save)
    show_flag = not args.no_show
    T = default_projection_matrix(4)

    ax = plot_unconstrained_simplex(T, show=show_flag)
    if args.save_figs:
        Path(paths.results_figures).mkdir(parents=True, exist_ok=True)
        (Path(paths.results_figures) / f"simplex{args.fig_suffix}.png").resolve()
        ax.figure.savefig(Path(paths.results_figures) / f"simplex{args.fig_suffix}.png", dpi=200)

    ax = plot_polytope(proj_vertices, common_limits=limits, show=show_flag)
    if args.save_figs:
        ax.figure.savefig(Path(paths.results_figures) / f"polytope{args.fig_suffix}.png", dpi=200)

    ax = plot_tetrahedral_subdivisions(proj_vertices, tri.simplices, common_limits=limits, show=show_flag)
    if args.save_figs:
        ax.figure.savefig(Path(paths.results_figures) / f"polytope_tetrahedra{args.fig_suffix}.png", dpi=200)

    # Save tables (optional)
    if args.save_csv:
        Path(paths.results_tables).mkdir(parents=True, exist_ok=True)
        df_vertices_x = pd.DataFrame(vertices_x, columns=[f"x{i+1}" for i in range(d)])
        df_proj_vertices = pd.DataFrame(proj_vertices, columns=["X", "Y", "Z"])
        df_tets = pd.DataFrame(
            {
                "tet_id": np.arange(tri.simplices.shape[0], dtype=int),
                "v0": tri.simplices[:, 0],
                "v1": tri.simplices[:, 1],
                "v2": tri.simplices[:, 2],
                "v3": tri.simplices[:, 3],
                "volume": tri.volumes,
                "weight": tri.volumes_norm,
            }
        )
        df_vertices_x.to_csv(Path(paths.results_tables) / "polytope_vertices_x.csv", index=False)
        df_proj_vertices.to_csv(Path(paths.results_tables) / "projected_vertices.csv", index=False)
        df_tets.to_csv(Path(paths.results_tables) / "tetrahedra_info.csv", index=False)
        print(f"[saved] {paths.results_tables}/polytope_vertices_x.csv")
        print(f"[saved] {paths.results_tables}/projected_vertices.csv")
        print(f"[saved] {paths.results_tables}/tetrahedra_info.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
