import numpy as np
from scipy.spatial import ConvexHull

#----------------------------
# Vertex enumeration / geometry helpers
#----------------------------

def compute_bounded_polytope_vertices_enumeration(L, U):
    """
    Enumerate extreme points of a bounded simplex slice via active constraints.

    Consider the feasible set S = { x ∈ R^n : sum(x)=1, L <= x <= U }.
    Let y = x - L. Then sum(y) = 1 - sum(L) and 0 <= y_i <= U_i - L_i.
    This routine enumerates candidate vertices in y-space by fixing (n-1)
    components to either their lower bound (0) or upper bound (U_i-L_i),
    then solving for the remaining free component. Feasible candidates are
    mapped back to x-space (x = y + L) and de-duplicated.

    Parameters
    ----------
    L : array-like of shape (n,)
        Per-component lower bounds with sum(L) <= 1.
    U : array-like of shape (n,)
        Per-component upper bounds with sum(U) >= 1 and U >= L elementwise.

    Returns
    -------
    vertices_x : ndarray of shape (n_vertices, n)
        Unique feasible vertices in the original x-space (on the simplex).
        May be empty if no candidates satisfy the bounds.

    Notes
    -----
    - Numerical feasibility checks use a small tolerance (≈1e-10).
    - This captures many, but not necessarily all, extreme points in degenerate
      cases; a convex-hull fallback is provided in `compute_bounded_polytope_vertices`.
    """
    total_mass = 1 - np.sum(L)  # Total mass available for y = x - L
    y_upper = U - L             # Upper bounds for y
    vertices_y = []             # List to hold vertices in y-space
    import itertools
    # Create constraint options for each component: each component can be fixed at lower ('L') or upper ('U')
    constraint_options = [(i, b) for i in range(len(L)) for b in ["L", "U"]]
    # For a DIM-dimensional design, choose (DIM-1) active constraints
    for active in itertools.combinations(constraint_options, len(L)-1):
        indices = [c[0] for c in active]
        # Skip if the chosen constraints do not cover distinct components
        if len(set(indices)) < (len(L)-1):
            continue
        # Initialize a candidate for y
        y_candidate = np.empty(len(L))
        y_candidate.fill(np.nan)
        # Set the active constraints:
        for idx, bound in active:
            y_candidate[idx] = 0.0 if bound == "L" else y_upper[idx]
        # Identify the free component and compute its value so that sum(y) = total_mass
        free_index = [i for i in range(len(L)) if i not in indices][0]
        fixed_sum = np.nansum(y_candidate)
        y_candidate[free_index] = total_mass - fixed_sum
        # Check if the candidate is feasible (within bounds)
        if np.all(y_candidate >= -1e-10) and np.all(y_candidate <= y_upper + 1e-10):
            vertices_y.append(y_candidate)
    if not vertices_y:
        return np.empty((0, len(L)))
    # Remove duplicates (within tolerance)
    unique = []
    for v in vertices_y:
        if not any(np.allclose(v, u, atol=1e-6) for u in unique):
            unique.append(v)
    unique = np.array(unique)
    # Map back to original x-space: x = y + L
    vertices_x = unique + L
    return vertices_x

#----------------------------
# Vertex computation with fallback
#----------------------------

def compute_bounded_polytope_vertices(L, U, n_samples=10000):
    """
    Compute extreme points of the bounded mixture polytope; fall back to hull.

    First attempts vertex enumeration via `compute_bounded_polytope_vertices_enumeration`.
    If no vertices are found (or enumeration is insufficient), draws Dirichlet samples
    within the simplex, filters by bounds, and returns the convex-hull vertices of the
    feasible set.

    Parameters
    ----------
    L : array-like of shape (n,)
        Lower bounds for mixture components.
    U : array-like of shape (n,)
        Upper bounds for mixture components.
    n_samples : int, optional
        Number of Dirichlet samples for the convex-hull fallback (default 10000).

    Returns
    -------
    vertices_x : ndarray of shape (n_vertices, n)
        Extreme points (vertices) of the feasible region in x-space.

    Raises
    ------
    ValueError
        If the fallback finds no feasible points under the supplied bounds.

    Notes
    -----
    The hull-based fallback approximates the true vertex set; increasing
    `n_samples` can improve coverage at additional computational cost.
    """
    vertices_x = compute_bounded_polytope_vertices_enumeration(L, U)
    if vertices_x.shape[0] == 0:
        samples = np.random.dirichlet(np.ones(len(L)), size=n_samples)
        mask = np.all((samples >= L - 1e-8) & (samples <= U + 1e-8), axis=1)
        feasible = samples[mask]
        if feasible.shape[0] == 0:
            raise ValueError("No feasible formulation samples found; check your bounds.")
        hull = ConvexHull(feasible)
        vertices_x = feasible[hull.vertices]
    return vertices_x

#----------------------------
# Linear projection
#----------------------------

def project_to_3d(vertices_x, T):
    """
    Apply a 4D→3D (or nD→3D) linear projection to vertex coordinates.

    Parameters
    ----------
    vertices_x : ndarray of shape (n_points, n)
        Points in the original mixture space.
    T : ndarray of shape (n, 3)
        Linear projection matrix mapping R^n → R^3.

    Returns
    -------
    proj_points : ndarray of shape (n_points, 3)
        Projected 3D coordinates.

    Notes
    -----
    This is a simple matrix product `vertices_x @ T`; no normalization or
    perspective effects are applied.
    """
    return np.dot(vertices_x, T)

#----------------------------
# Tetrahedral volume
#----------------------------

def compute_tetrahedron_volume(verts):
    """
    Compute the volume of a tetrahedron from its four 3D vertices.

    The volume is |det([v1−v0, v2−v0, v3−v0])| / 6, where v0..v3 ∈ R^3.

    Parameters
    ----------
    verts : ndarray of shape (4, 3)
        Four 3D vertices defining the tetrahedron.

    Returns
    -------
    volume : float
        Nonnegative tetrahedral volume.

    Notes
    -----
    Geometrically degenerate inputs (coplanar vertices) yield near-zero volume
    subject to floating-point precision.
    """
    return np.abs(np.linalg.det(np.column_stack([verts[1]-verts[0],
                                                   verts[2]-verts[0],
                                                   verts[3]-verts[0]]))) / 6
