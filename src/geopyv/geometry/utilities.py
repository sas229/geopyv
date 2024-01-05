"""

Utilities module for geopyv.

"""
import numpy as np


def area_to_length(area):
    """

    Function that returns a characteristic length given
    an element area, based on an equilateral triangle.

    Parameters
    ----------
    area : float
        Element area.


    Returns
    -------
    length : float
        Characteristic length.

    """
    length = np.sqrt(4 * abs(area) / np.sqrt(3))
    return length


def plot_triangulation(elements, x, y, mesh_order):
    """

    Method to compute a first order triangulation from a
    second order element connectivity array and coordinates.

    Parameters
    ----------
    elements : np.ndarray (Nx, 6)
        Element connectivity array.
    x : np.ndarray (Nx, 1)
        Horizontal coordinate array.
    y : np.ndarray (Nx, 1)
        Vertical coordinate array.


    Returns
    -------
    mesh_triangulation : np.ndarray (Nx, 7)
        Mesh triangulation array for plot purposes forming
        closed triangles.
    x_p : np.ndarray (Nx, 1)
        Horizontal coordinate of triangle vertices.
    y_p : np.ndarray (Nx, 1)
        Vertical coordinate of triangle vertices.

    """
    if mesh_order == 1:
        x_p = x[elements[:, [0, 1, 2, 0]]]
        y_p = y[elements[:, [0, 1, 2, 0]]]
        mesh_triangulation = elements
    elif mesh_order == 2:
        x_p = x[elements[:, [0, 3, 1, 4, 2, 5, 0]]]
        y_p = y[elements[:, [0, 3, 1, 4, 2, 5, 0]]]
        mesh_triangulation = elements[
            :, [[0, 3, 5], [1, 3, 4], [2, 4, 5], [3, 4, 5]]
        ].reshape(-1, 3)

    return mesh_triangulation, x_p, y_p


def PolyArea(pts):
    """

    A function that returns the area of the input polygon.

    Parameters
    ----------
    pts : `numpy.ndarray` (Nx,2)
        Clockwise/anti-clockwise ordered coordinates.

    """
    x = pts[:, 0]
    y = pts[:, 1]
    return abs(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def polysect(n):
    """
    Function which iterates through the segments of a polygon checking for
    intersection. Assumes input is clockwise/anti-clockwise ordered.

    Returns
    -------
    True : Two (or more) segments intersect.
    False: No intersection (spare that at nodes).
    """
    co = np.asarray(
        [[0, 2], [0, 3], [0, 4], [1, 3], [1, 4], [1, 5], [2, 4], [2, 5], [3, 5]]
    )
    for i in range(len(co)):
        if intersect(
            n[co[i, 0] % 6],
            n[(co[i, 0] + 1) % 6],
            n[co[i, 1] % 6],
            n[(co[i, 1] + 1) % 6],
        ):
            return co[i]
    return False


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def polycentroid(coords):
    centroid = np.zeros(2)
    sa = 0
    for i in range(len(coords)):
        j = (i + 1) % len(coords)
        a = (coords[i, 0] * coords[j, 1]) - (coords[j, 0] * coords[i, 1])
        sa += a
        centroid[0] += (coords[i, 0] + coords[j, 0]) * a
        centroid[1] += (coords[i, 0] + coords[j, 1]) * a
    centroid /= 3 * sa

    return centroid
