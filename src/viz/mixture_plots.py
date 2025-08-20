import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------------------------------
# Helpers
# ---------------------------------
def _finalize(fig, outpath=None, dpi=300, show=False):
    """Save/show/close a figure in a consistent way."""
    if outpath is not None:
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight", transparent=True)
    if show:
        plt.show()
    else:
        plt.close(fig)

#---------------------------------
# 3D Visualization Utilities
#---------------------------------

def get_common_axis_limits(proj_vertices):
    """
    Compute tight axis limits for a 3D point cloud.

    Parameters
    ----------
    proj_vertices : ndarray of shape (n_points, 3)
        3D coordinates (X, Y, Z) of the projected vertices.

    Returns
    -------
    limits : tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
        A tuple of per-axis limits: ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
    """
    x_lim = (proj_vertices[:,0].min(), proj_vertices[:,0].max())
    y_lim = (proj_vertices[:,1].min(), proj_vertices[:,1].max())
    z_lim = (proj_vertices[:,2].min(), proj_vertices[:,2].max())
    return x_lim, y_lim, z_lim

def set_axes_limits(ax, limits):
    """
    Apply precomputed axis limits to a 3D Matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        Target 3D axis.
    limits : tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
        Per-axis limits as returned by `get_common_axis_limits`:
        ((xmin, xmax), (ymin, ymax), (zmin, zmax)).

    Returns
    -------
    None
        Limits are applied in-place to `ax`.
    """
    x_lim, y_lim, z_lim = limits
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)

#---------------------------------
# 3D Plotting: Simplex and Polytope
#---------------------------------

def plot_unconstrained_simplex(T, outpath=None, dpi=300, show=False):
    """
    Plot the standard 4-component simplex projected to 3D.

    The function takes the 4D standard basis (identity matrix) and projects
    it using the provided transformation matrix `T` (shape (4, 3)). This
    visualizes the unconstrained simplex corners in 3D.

    Parameters
    ----------
    T : ndarray of shape (4, 3)
        Linear projection matrix from the 4D mixture space to 3D.
    outpath : str or Path, optional
        If provided, save the figure to this path (e.g., PDF).
    dpi : int, optional
        Resolution for saving the figure (default 300).
    show : bool, optional
        If True, display the figure window; otherwise close after saving.

    Returns
    -------
    None
    """
    simplex_vertices = np.eye(4)
    proj_simplex = np.dot(simplex_vertices, T)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(proj_simplex[:,0], proj_simplex[:,1], proj_simplex[:,2],
               color="blue", s=50, label="Simplex Vertices")
    edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    for edge in edges:
        p1 = proj_simplex[edge[0]]
        p2 = proj_simplex[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="black")
    ax.set_title("Unconstrained 4-Component Simplex (Projected to 3D)")
    ax.legend()
    _finalize(fig, outpath, dpi, show)

#---------------------------------
# 3D Plotting: Polytope Outline
#---------------------------------

def plot_polytope(proj_vertices, common_limits, outpath=None, dpi=300, show=False):
    """
    Draw the convex hull (polytope) from projected vertices.

    Parameters
    ----------
    proj_vertices : ndarray of shape (n_points, 3)
        3D coordinates of the projected polytope vertices.
    common_limits : tuple
        Axis limits to apply (see `get_common_axis_limits`).
    outpath : str or Path, optional
        If provided, save the figure to this path.
    dpi : int, optional
        Resolution for saving (default 300).
    show : bool, optional
        If True, display the figure window.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    hull = ConvexHull(proj_vertices)
    for s in hull.simplices:
        s_closed = np.append(s, s[0])
        ax.plot(proj_vertices[s_closed,0],
                proj_vertices[s_closed,1],
                proj_vertices[s_closed,2],
                color="black")
    ax.scatter(proj_vertices[:,0], proj_vertices[:,1],
               proj_vertices[:,2], color="blue", s=50, label="Polytope Vertices")
    ax.set_title("Bounded Polytope (Projected to 3D)")
    set_axes_limits(ax, common_limits)
    ax.legend()
    _finalize(fig, outpath, dpi, show)

#---------------------------------
# 3D Plotting: Tetrahedral Decomposition
#---------------------------------

def plot_tetrahedral_subdivisions(proj_vertices, simplices, common_limits, outpath=None, dpi=300, show=False):
    """
    Visualize a tetrahedral subdivision of the polytope.

    Parameters
    ----------
    proj_vertices : ndarray of shape (n_points, 3)
        3D coordinates of the projected polytope vertices.
    simplices : ndarray of shape (n_tetra, 4)
        Indices (into `proj_vertices`) defining each tetrahedron (e.g., from
        `scipy.spatial.Delaunay(...).simplices`).
    common_limits : tuple
        Axis limits to apply (see `get_common_axis_limits`).
    outpath : str or Path, optional
        If provided, save the figure to this path.
    dpi : int, optional
        Resolution for saving (default 300).
    show : bool, optional
        If True, display the figure window.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.viridis(np.linspace(0,1,len(simplices)))
    for i, simplex in enumerate(simplices):
        tet_pts = proj_vertices[simplex]
        faces = [
            [tet_pts[0], tet_pts[1], tet_pts[2]],
            [tet_pts[0], tet_pts[1], tet_pts[3]],
            [tet_pts[0], tet_pts[2], tet_pts[3]],
            [tet_pts[1], tet_pts[2], tet_pts[3]]
        ]
        poly = Poly3DCollection(faces, alpha=0.3)
        poly.set_facecolor(colors[i])
        poly.set_edgecolor('k')
        ax.add_collection3d(poly)
    hull = ConvexHull(proj_vertices)
    for s in hull.simplices:
        s_closed = np.append(s, s[0])
        ax.plot(proj_vertices[s_closed,0],
                proj_vertices[s_closed,1],
                proj_vertices[s_closed,2],
                color="black", lw=1)
    ax.scatter(proj_vertices[:,0], proj_vertices[:,1],
               proj_vertices[:,2], color="blue", s=50, label="Vertices")
    ax.set_title("Polytope with Tetrahedral Subdivisions")
    set_axes_limits(ax, common_limits)
    ax.legend()
    _finalize(fig, outpath, dpi, show)

#---------------------------------
# 3D Plotting: Polytope + Samples
#---------------------------------

def plot_polytope_with_samples(proj_vertices, samples, title, common_limits, outpath=None, dpi=300, show=False):
    """
    Overlay sample points on the convex hull of the projected polytope.

    Parameters
    ----------
    proj_vertices : ndarray of shape (n_points, 3)
        3D coordinates of the projected polytope vertices.
    samples : ndarray of shape (n_samples, 3)
        3D sample locations to overlay.
    title : str
        Title of the plot.
    common_limits : tuple
        Axis limits to apply (see `get_common_axis_limits`).
    outpath : str or Path, optional
        If provided, save the figure to this path.
    dpi : int, optional
        Resolution for saving (default 300).
    show : bool, optional
        If True, display the figure window.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    hull = ConvexHull(proj_vertices)
    for s in hull.simplices:
        s_closed = np.append(s, s[0])
        ax.plot(proj_vertices[s_closed,0],
                proj_vertices[s_closed,1],
                proj_vertices[s_closed,2],
                color="black")
    ax.scatter(proj_vertices[:,0], proj_vertices[:,1],
               proj_vertices[:,2], color="blue", s=50, label="Vertices")
    ax.scatter(samples[:,0], samples[:,1],
               samples[:,2], color="red", s=20, alpha=0.6, label="Samples")
    ax.set_title(title)
    set_axes_limits(ax, common_limits)
    ax.legend()
    _finalize(fig, outpath, dpi, show)

#---------------------------------
# 3D Plotting: Subdivisions + Samples
#---------------------------------

def plot_subdivisions_with_samples(proj_vertices, simplices, samples, common_limits, outpath=None, dpi=300, show=False):
    """
    Overlay sample points on a tetrahedral subdivision of the polytope.

    Parameters
    ----------
    proj_vertices : ndarray of shape (n_points, 3)
        3D coordinates of the projected polytope vertices.
    simplices : ndarray of shape (n_tetra, 4)
        Indices defining tetrahedra (e.g., from Delaunay). Each row contains
        4 vertex indices into `proj_vertices`.
    samples : ndarray of shape (n_samples, 3)
        3D sample locations to overlay.
    common_limits : tuple
        Axis limits to apply (see `get_common_axis_limits`).
    outpath : str or Path, optional
        If provided, save the figure to this path.
    dpi : int, optional
        Resolution for saving (default 300).
    show : bool, optional
        If True, display the figure window.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.viridis(np.linspace(0,1,len(simplices)))
    for i, simplex in enumerate(simplices):
        tet_pts = proj_vertices[simplex]
        faces = [
            [tet_pts[0], tet_pts[1], tet_pts[2]],
            [tet_pts[0], tet_pts[1], tet_pts[3]],
            [tet_pts[0], tet_pts[2], tet_pts[3]],
            [tet_pts[1], tet_pts[2], tet_pts[3]]
        ]
        poly = Poly3DCollection(faces, alpha=0.3)
        poly.set_facecolor(colors[i])
        poly.set_edgecolor('k')
        ax.add_collection3d(poly)
    hull = ConvexHull(proj_vertices)
    for s in hull.simplices:
        s_closed = np.append(s, s[0])
        ax.plot(proj_vertices[s_closed,0],
                proj_vertices[s_closed,1],
                proj_vertices[s_closed,2],
                color="black", lw=1)
    ax.scatter(proj_vertices[:,0], proj_vertices[:,1],
               proj_vertices[:,2], color="blue", s=50, label="Vertices")
    ax.scatter(samples[:,0], samples[:,1],
               samples[:,2], color="red", s=20, alpha=0.6, label="Samples")
    ax.set_title("Tetrahedral Subdivisions with Samples")
    set_axes_limits(ax, common_limits)
    ax.legend()
    _finalize(fig, outpath, dpi, show)

#---------------------------------
# 3D Plotting: Polytope + DoE Design
#---------------------------------

def plot_polytope_with_doe(proj_vertices, doe_proj, common_limits, outpath=None, dpi=300, show=False):
    """
    Overlay a Design-of-Experiments (DoE) point set on the polytope.

    Parameters
    ----------
    proj_vertices : ndarray of shape (n_points, 3)
        3D coordinates of the projected polytope vertices.
    doe_proj : ndarray of shape (n_doe, 3)
        3D coordinates of DoE design points already projected into the same
        space as `proj_vertices`.
    common_limits : tuple
        Axis limits to apply (see `get_common_axis_limits`).
    outpath : str or Path, optional
        If provided, save the figure to this path.
    dpi : int, optional
        Resolution for saving (default 300).
    show : bool, optional
        If True, display the figure window.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    hull = ConvexHull(proj_vertices)
    for s in hull.simplices:
        s_closed = np.append(s, s[0])
        ax.plot(proj_vertices[s_closed,0],
                proj_vertices[s_closed,1],
                proj_vertices[s_closed,2],
                color="black")
    ax.scatter(proj_vertices[:,0], proj_vertices[:,1],
               proj_vertices[:,2], color="blue", s=50, label="Polytope Vertices")
    ax.scatter(doe_proj[:,0], doe_proj[:,1],
               doe_proj[:,2], color="green", s=30, alpha=0.8, label="DOE Design")
    ax.set_title("Polytope with DOE Design")
    set_axes_limits(ax, common_limits)
    ax.legend()
    _finalize(fig, outpath, dpi, show)