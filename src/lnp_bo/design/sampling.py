import numpy as np
from scipy.stats import qmc

##################
# Sobol Sampling #
##################


def generate_sobol_samples(n_samples, d=3):
    """
    Generate a Sobol sequence of points in the unit cube [0,1]^d.
    
    Parameters
    ----------
    n_samples : int
        Number of Sobol points to generate.
    d : int, optional
        Dimension of the Sobol points (default is 3).
    
    Returns
    -------
    samples : ndarray, shape (n_samples, d)
        A set of Sobol points.
    """
    sobol_engine = qmc.Sobol(d=d, scramble=True)
    m = int(np.ceil(np.log2(n_samples)))
    return sobol_engine.random_base2(m=m)[:n_samples]

def generate_1d_sobol_samples(n_samples, lower, upper):
    """
    Generate a 1D Sobol sequence scaled to a given interval.
    
    Parameters
    ----------
    n_samples : int
        Number of 1D Sobol points to generate.
    lower : float
        Lower bound of the interval.
    upper : float
        Upper bound of the interval.
        
    Returns
    -------
    samples : ndarray, shape (n_samples,)
        A set of Sobol points in [lower, upper].
    """
    sobol_engine = qmc.Sobol(d=1, scramble=True)
    m = int(np.ceil(np.log2(n_samples)))
    samples = sobol_engine.random_base2(m=m)[:n_samples].flatten()
    return lower + samples * (upper - lower)

def sobol_to_bary(u):
    """
    Convert a point in [0,1]^3 to barycentric coordinates for a tetrahedron.
    
    Uses a stick-breaking transformation to transform the 3D point u into
    barycentric coordinates that sum to 1.
    
    Parameters
    ----------
    u : array-like, shape (3,)
        A point in the unit cube.
        
    Returns
    -------
    bary : ndarray, shape (4,)
        Barycentric coordinates corresponding to u.
    """
    u1, u2, u3 = u
    b1 = 1 - u1**(1/3)
    b2 = u1**(1/3) * (1 - u2**(1/2))
    b3 = u1**(1/3) * u2**(1/2) * (1 - u3)
    b4 = u1**(1/3) * u2**(1/2) * u3
    return np.array([b1, b2, b3, b4])

def normalize_to_unit_cube(points):
    """
    Normalize points to the unit cube [0,1]^d.
    
    This function scales an array of points so that each coordinate is mapped
    to the interval [0,1] based on the minimum and maximum values in that dimension.
    
    Parameters
    ----------
    points : ndarray, shape (n_points, d)
        Input points.
        
    Returns
    -------
    normalized_points : ndarray, shape (n_points, d)
        Points normalized to [0,1]^d.
    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return (points - mins) / (maxs - mins + 1e-12)

def star_discrepancy(points, grid_points=30):
    """
    Approximate the star discrepancy of a set of points in [0,1]^d.
    
    The star discrepancy measures the uniformity of the point set. This function
    evaluates the discrepancy on a grid of test points.
    
    Parameters
    ----------
    points : ndarray, shape (n_points, d)
        The point set, assumed to lie in [0,1]^d.
    grid_points : int, optional
        Number of grid points per dimension (default is 30).
        
    Returns
    -------
    discrepancy : float
        The maximum absolute difference between the empirical distribution function
        and the uniform measure over the grid.
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