import numpy as np
from scipy.spatial import Delaunay, ConvexHull

###################################################################
# Geometric design of the bounded polytope for the mixture design #
###################################################################

def compute_bounded_polytope_vertices_enumeration(L, U):
    """
    Enumerate the vertices of the bounded mixture design in the shifted space.
    
    In mixture design, if x is a formulation vector (that sums to 1) with lower bounds L 
    and upper bounds U, then by defining y = x - L we have sum(y) = 1 - sum(L). The feasible 
    region for y is a subset of the standard simplex with upper bounds y[i] <= U[i] - L[i].
    
    This function enumerates vertices of that region by fixing (DIM - 1) components at either 
    their lower bound (0) or their upper bound (U[i]-L[i]) and solving for the remaining component.
    
    Parameters
    ----------
    L : array-like, shape (DIM,)
        Lower bounds for the formulation components.
    U : array-like, shape (DIM,)
        Upper bounds for the formulation components.
        
    Returns
    -------
    vertices_x : ndarray, shape (n_vertices, DIM)
        The vertices of the bounded mixture design in the original x-space.
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

def compute_bounded_polytope_vertices(L, U, n_samples=10000):
    """
    Compute the vertices of the bounded formulation design.
    
    This function first attempts to enumerate the vertices via the above method.
    If no vertices are found (or if the enumeration is insufficient), it falls back on
    Dirichlet sampling and extracts the convex hull of the feasible samples.
    
    Parameters
    ----------
    L : array-like, shape (DIM,)
        Lower bounds for formulation.
    U : array-like, shape (DIM,)
        Upper bounds for formulation.
    n_samples : int, optional
        Number of Dirichlet samples to generate if enumeration fails.
    
    Returns
    -------
    vertices_x : ndarray, shape (n_vertices, DIM)
        Vertices of the bounded formulation design.
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

def project_to_3d(vertices_x, T):
    """
    Project points from DIM-dimensional space to 3D.
    
    This function uses a provided transformation matrix T (of shape (DIM, 3))
    to project the original design points to 3D for visualization.
    
    Parameters
    ----------
    vertices_x : ndarray, shape (n_points, DIM)
        The original design points in DIM-dimensional space.
    T : ndarray, shape (DIM, 3)
        Transformation matrix for projecting points to 3D.
        
    Returns
    -------
    proj_points : ndarray, shape (n_points, 3)
        The projected 3D points.
    """
    return np.dot(vertices_x, T)

def compute_tetrahedron_volume(verts):
    """
    Compute the volume of a tetrahedron.
    
    Given four 3D vertices (an array of shape (4,3)), this function computes
    the volume using the determinant formula:
    
        Volume = |det([v1-v0, v2-v0, v3-v0])| / 6
    
    Parameters
    ----------
    verts : ndarray, shape (4, 3)
        Four vertices of a tetrahedron in 3D.
        
    Returns
    -------
    volume : float
        The volume of the tetrahedron.
    """
    return np.abs(np.linalg.det(np.column_stack([verts[1]-verts[0],
                                                   verts[2]-verts[0],
                                                   verts[3]-verts[0]]))) / 6