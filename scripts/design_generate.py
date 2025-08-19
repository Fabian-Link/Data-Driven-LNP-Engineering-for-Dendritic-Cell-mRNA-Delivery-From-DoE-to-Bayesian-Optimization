# scripts/design_generate.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from lnp_bo.io.config import load_bounds, load_paths
from lnp_bo.design.mixture_polytope import build_projected_triangulated_polytope
from lnp_bo.design.sampling import (
    choose_best_sobol_replica,
    random_baseline_samples,
    normalize_to_unit_cube,
    star_discrepancy,
    append_1d_sobol_column,
)
from lnp_bo.viz.mixture_plots import (
    get_common_axis_limits,
    plot_polytope,
    plot_polytope_with_samples,
    plot_subdivisions_with_samples,
    plot_polytope_with_doe,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate initial space-filling designs inside a bounded mixture polytope using Sobol sequences."
    )
    p.add_argument("--bounds", default="configs/bounds.yaml", help="Path to bounds.yaml")
    p.add_argument("--paths", default="configs/paths.yaml", help="Path to paths.yaml")
    p.add_argument("--n-samples", type=int, default=8, help="Number of design points")
    p.add_argument("--replicates", type=int, default=100, help="Number of Sobol replicates to screen")
    p.add_argument("--seed", type=int, default=None, help="Base seed for reproducibility")
    p.add_argument("--include-random", action="store_true", help="Also generate a uniform-random baseline set")
    p.add_argument("--doe-csv", type=str, default=None, help="Optional path to a DoE CSV (columns: the 4 mixture parts)")
    p.add_argument("--save-csv", action="store_true", help="Save design CSVs to results/tables")
    p.add_argument("--save-figs", action="store_true", help="Save figures to results/figures")
    p.add_argument("--export-excel", action="store_true", help="Export final 5D (mix + N/P) to Excel (paths.artifacts.design_export)")
    p.add_argument("--no-show", action="store_true", help="Do not display plots (useful for CI)")
    p.add_argument("--fig-suffix", default="", help="Optional suffix for figure filenames, e.g. _v1")
    return p.parse_args()


def maybe_load_doe(path_hint: Path | None, n_parts: int) -> np.ndarray | None:
    if path_hint is None:
        return None
    if not path_hint.exists():
        return None
    try:
        df = pd.read_csv(path_hint)
    except Exception:
        try:
            df = pd.read_excel(path_hint)
        except Exception:
            return None
    # Use first n_parts numeric columns
    num = df.select_dtypes(include=["number"])
    if num.shape[1] < n_parts:
        return None
    X = num.iloc[:, :n_parts].to_numpy(float)
    # filter NaNs/inf
    mask = np.isfinite(X).all(axis=1)
    return X[mask]


def main() -> int:
    args = parse_args()

    # Load configs
    bounds = load_bounds(args.bounds)
    paths = load_paths(args.paths)

    labels = list(bounds.labels)
    L = np.asarray(bounds.lower, dtype=float)
    U = np.asarray(bounds.upper, dtype=float)
    d = L.size
    if d != 4:
        print(f"[error] This script assumes a 4-part mixture (got d={d}). Adjust code for general d.", file=sys.stderr)
        return 2

    # Build the polytope (vertices in x-space), project to 3D, and triangulate
    vertices_x, proj_vertices, tri = build_projected_triangulated_polytope(L, U)
    limits = get_common_axis_limits(proj_vertices)

    print("=== Design-space ===")
    print(f"parts (d):           {d}")
    print(f"# vertices:          {vertices_x.shape[0]}")
    print(f"# tetrahedra:        {tri.simplices.shape[0]}")
    print(f"volume weights sum:  {float(tri.volumes_norm.sum()):.6f}")

    # 1) Best Sobol replicate (by star discrepancy on normalized 3D)
    sob = choose_best_sobol_replica(
        n_samples=args.n_samples,
        n_replicates=args.replicates,
        proj_vertices=proj_vertices,
        vertices_x=vertices_x,
        simplices=tri.simplices,
        volumes_norm=tri.volumes_norm,
        grid_points=30,
        base_seed=args.seed,
    )
    sobol_3d = sob.points_3d
    sobol_4d = sob.points_d
    sobol_disc = sob.discrepancy

    print("\n=== Sobol selection ===")
    print(f"replicates screened: {args.replicates}")
    print(f"best discrepancy:    {sobol_disc:.6f}")
    if sob.seed is not None:
        print(f"best seed:           {sob.seed}")

    # 2) Random baseline (optional)
    rand_3d = rand_4d = None
    disc_rand = None
    if args.include_random:
        rng = np.random.default_rng(args.seed)
        rnd = random_baseline_samples(
            n_samples=args.n_samples,
            proj_vertices=proj_vertices,
            vertices_x=vertices_x,
            simplices=tri.simplices,
            volumes_norm=tri.volumes_norm,
            rng=rng,
        )
        rand_3d = rnd.points_3d
        rand_4d = rnd.points_d
        disc_rand = star_discrepancy(normalize_to_unit_cube(rand_3d), grid_points=30)
        print("\n=== Random baseline ===")
        print(f"discrepancy:         {disc_rand:.6f}")

    # 3) DoE overlay (optional path or default to data/raw/doe_design.csv if present)
    doe_arr = None
    doe_path = None
    if args.doe_csv:
        doe_path = Path(args.doe_csv)
    else:
        # default hint under data/raw
        doe_path = Path(paths.data_raw) / "doe_design.csv"
    doe_arr = maybe_load_doe(doe_path if doe_path else None, n_parts=d)
    if doe_arr is not None:
        # project to 3D for plotting & compute discrepancy
        H = doe_arr @ (proj_vertices.T @ np.linalg.pinv(vertices_x.T)).T if False else None  # placeholder if needed
        doe_proj = doe_arr @ np.linalg.lstsq(vertices_x, proj_vertices, rcond=None)[0]  # linear map from x to proj
        doe_disc = star_discrepancy(normalize_to_unit_cube(doe_proj), grid_points=30)
        print("\n=== DoE (loaded) ===")
        print(f"path:                {str(doe_path)}")
        print(f"rows:                {doe_arr.shape[0]}")
        print(f"discrepancy (3D):    {doe_disc:.6f}")
    else:
        doe_proj = None

    # 4) Append 1D Sobol N/P column to Sobol 4D -> make 5D
    sobol_5d = append_1d_sobol_column(
        sobol_4d,
        lower=bounds.np_lower,
        upper=bounds.np_upper,
        seed=args.seed,
    )

    # DataFrames with friendly column names
    df_sobol4 = pd.DataFrame(sobol_4d, columns=labels)
    df_sobol5 = pd.DataFrame(np.hstack([sobol_4d, sobol_5d[:, [-1]]]), columns=labels + ["N/P"])

    if rand_4d is not None:
        df_rand4 = pd.DataFrame(rand_4d, columns=labels)

    # Feasibility checks
    sums = df_sobol4.sum(axis=1).values
    print("\n=== Feasibility checks (Sobol 4D) ===")
    print(f"sum(x) min/max:      {sums.min():.6f} / {sums.max():.6f}")
    print(f"within [L,U] all?:   {bool(((sobol_4d >= L - 1e-9) & (sobol_4d <= U + 1e-9)).all())}")

    # 5) Export
    if args.save_csv:
        Path(paths.results_tables).mkdir(parents=True, exist_ok=True)
        df_sobol4.to_csv(Path(paths.results_tables) / "best_sobol_sampled_points_4D.csv", index=False)
        df_sobol5.to_csv(Path(paths.results_tables) / "best_sobol_sampled_points_5D.csv", index=False)
        if rand_4d is not None:
            df_rand4.to_csv(Path(paths.results_tables) / "random_sampled_points_4D.csv", index=False)
        # Also save vertices/triangulation? Reserved for build_polytope.py
        print(f"[saved] {paths.results_tables}/best_sobol_sampled_points_4D.csv")
        print(f"[saved] {paths.results_tables}/best_sobol_sampled_points_5D.csv")
        if rand_4d is not None:
            print(f"[saved] {paths.results_tables}/random_sampled_points_4D.csv")

    if args.export_excel:
        out_xlsx = Path(paths.design_export)
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)
        df_sobol5.to_excel(out_xlsx, index=False)
        print(f"[exported] Excel design → {out_xlsx}")

    # 6) Plots
    show_flag = not args.no_show
    ax = plot_polytope(proj_vertices, common_limits=limits, show=show_flag)
    if args.save_figs:
        Path(paths.results_figures).mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(Path(paths.results_figures) / f"polytope{args.fig_suffix}.png", dpi=200)

    ax = plot_polytope_with_samples(proj_vertices, sobol_3d, "Polytope with Sobol Samples",
                                    common_limits=limits, show=show_flag)
    if args.save_figs:
        ax.figure.savefig(Path(paths.results_figures) / f"polytope_sobol_samples{args.fig_suff ix}.png", dpi=200)

    ax = plot_subdivisions_with_samples(proj_vertices, tri.simplices, sobol_3d,
                                        common_limits=limits, show=show_flag)
    if args.save_figs:
        ax.figure.savefig(Path(paths.results_figures) / f"polytope_tetrahedra_sobol{args.fig_suffix}.png", dpi=200)

    if rand_3d is not None:
        ax = plot_polytope_with_samples(proj_vertices, rand_3d, "Polytope with Random Samples",
                                        common_limits=limits, show=show_flag)
        if args.save_figs:
            ax.figure.savefig(Path(paths.results_figures) / f"polytope_random_samples{args.fig_suffix}.png", dpi=200)

    if doe_proj is not None:
        ax = plot_polytope_with_doe(proj_vertices, doe_proj, common_limits=limits, show=show_flag)
        if args.save_figs:
            ax.figure.savefig(Path(paths.results_figures) / f"polytope_doe{args.fig_suffix}.png", dpi=200)

    # Final console summary
    print("\n=== Summary ===")
    print(f"Sobol discrepancy:    {sobol_disc:.6f}")
    if disc_rand is not None:
        print(f"Random discrepancy:   {disc_rand:.6f}")
    print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
