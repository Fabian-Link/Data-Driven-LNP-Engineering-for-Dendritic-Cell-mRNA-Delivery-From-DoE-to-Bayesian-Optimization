import numpy as np
from scipy.stats import qmc

#----------------------------
# Sobol Sequence Sampling
#----------------------------

def generate_sobol_samples(n_samples, d=3):
    """
    Generate n Sobol low-discrepancy points in the unit hypercube [0, 1]^d.

    Parameters
    ----------
    n_samples : int
        Number of points to return.
    d : int, optional
        Dimensionality of the points (default 3).

    Returns
    -------
    samples : ndarray, shape (n_samples, d)
        Sobol sequence points in [0, 1]^d.

    Notes
    -----
    Uses `qmc.Sobol(scramble=True)` and draws `2**m` points with
    `m = ceil(log2(n_samples))`, then truncates to `n_samples`. This preserves
    Sobol structure while allowing arbitrary `n_samples`.
    """
    sobol_engine = qmc.Sobol(d=d, scramble=True)
    m = int(np.ceil(np.log2(n_samples)))
    return sobol_engine.random_base2(m=m)[:n_samples]

#----------------------------
# 1-D Sobol Sampling
#----------------------------

def generate_1d_sobol_samples(n_samples, lower, upper):
    """
    Generate n 1-D Sobol points and linearly scale them to [lower, upper].

    Parameters
    ----------
    n_samples : int
        Number of points to return.
    lower : float
        Inclusive lower bound of the target interval.
    upper : float
        Inclusive upper bound of the target interval.

    Returns
    -------
    samples : ndarray, shape (n_samples,)
        1-D Sobol points mapped to [lower, upper].

    Notes
    -----
    This is a convenience wrapper around `generate_sobol_samples` with `d=1`
    followed by an affine transform to the requested interval.
    """
    sobol_engine = qmc.Sobol(d=1, scramble=True)
    m = int(np.ceil(np.log2(n_samples)))
    samples = sobol_engine.random_base2(m=m)[:n_samples].flatten()
    return lower + samples * (upper - lower)

#-------------------------------------------------
# Transform euclidean to barycentric coordinates
#-------------------------------------------------

def sobol_to_bary(u):
    """
    Map a 3-D unit-cube point to 4-simplex barycentric coordinates.

    Uses a stick-breaking transformation so the returned vector is nonnegative
    and sums to 1 (i.e., valid barycentric weights for a tetrahedron).

    Parameters
    ----------
    u : array-like, shape (3,)
        Point in [0, 1]^3.

    Returns
    -------
    bary : ndarray, shape (4,)
        Barycentric coordinates (b1, b2, b3, b4) with sum(bi) = 1.

    Notes
    -----
    This transformation corresponds to sampling from a Dirichlet(1,1,1,1)
    via inverse CDF of the stick-breaking process, but here we deterministically
    transform a given `u`.
    """
    u1, u2, u3 = u
    b1 = 1 - u1**(1/3)
    b2 = u1**(1/3) * (1 - u2**(1/2))
    b3 = u1**(1/3) * u2**(1/2) * (1 - u3)
    b4 = u1**(1/3) * u2**(1/2) * u3
    return np.array([b1, b2, b3, b4])

#----------------------------
# Normalization
#----------------------------

def normalize_to_unit_cube(points):
    """
    Min-max scale each column of X to [0, 1].

    Parameters
    ----------
    points : ndarray, shape (n_points, d)
        Input array to normalize.

    Returns
    -------
    normalized_points : ndarray, shape (n_points, d)
        Column-wise normalized data in [0, 1].

    Notes
    -----
    If a column is constant, the denominator would be zero; an `1e-12`
    epsilon is added to avoid division by zero.
    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return (points - mins) / (maxs - mins + 1e-12)

#----------------------------
# Calculate Star Discrepancy
#----------------------------

def star_discrepancy(points, grid_points=30):
    """
    Approximate the star discrepancy of points in [0, 1]^d via a uniform grid.

    The star discrepancy measures uniformity by comparing the empirical CDF
    within axis-aligned boxes anchored at the origin to the true volume.

    Parameters
    ----------
    points : ndarray, shape (n_points, d)
        Point set assumed to lie in [0, 1]^d.
    grid_points : int, optional
        Number of grid values per axis used for evaluation (default 30).

    Returns
    -------
    discrepancy : float
        Max absolute deviation between empirical mass and box volume over the grid.

    Notes
    -----
    This is a coarse grid-based approximation with complexity
    O(grid_points**d * n_points) and becomes expensive for large `d`.
    """
    N, d = points.shape
    grid = np.linspace(0, 1, grid_points)
    mesh = np.meshgrid(*([grid]*d))
    test_pts = np.column_stack([m.flatten() for m in mesh])
    disc_vals = []
    for r in test_pts:
        count = np.sum(np.all(points <= r, axis=1))
        empirical = count / N
        volume = np.prod(r)
        disc_vals.append(np.abs(empirical - volume))
    return np.max(disc_vals)
