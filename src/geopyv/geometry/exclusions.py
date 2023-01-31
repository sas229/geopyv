"""

Exclusions module for geopyv.

"""
import numpy as np


def circular_exclusion(coord, radius, size):
    """

    Function to define an array of circular exclusion coordinates.

    Parameters
    ----------
    coord : `numpy.ndarray` (x, y)
        Coordinate of exclusion in pixels.
    radius : int
        Radius of exclusion in pixels.
    size : int
        Required subset spacing on exclusion boundary in pixels.


    Returns
    -------
    exclusion : np.ndarray (Nx,Ny)
        Array of coordinates defining the exclusion.

    """
    if type(coord) != np.ndarray:
        raise TypeError(
            "Coordinate of exclusion of invalid type. Must of `numpy.ndarray` type."
        )
    elif type(radius) != int:
        radius = int(radius)
    elif int(radius) < 0:
        raise ValueError(
            "Radius of circular exclusion invalid. Must be positive integer."
        )
    number_points = np.maximum(
        6, int(2 * np.pi * radius / size)
    )  # Use a minimum of six points irrespective of target size defined.
    theta = np.linspace(0, 2 * np.pi, number_points, endpoint=False)
    x = radius * np.cos(theta) + coord[0]
    y = radius * np.sin(theta) + coord[1]
    exclusion = np.column_stack((x, y))

    return exclusion


def circular_exclusion_list(coords, radius, size):
    """

    Function to define a list of circular exclusion coordinates.


    Parameters
    ----------
    coord : np.ndarray (x,y)
        Coordinate of exclusion in pixels.
    radius : int
        Radius of exclusion in pixels.
    size : int
        Required subset spacing on exclusion boundary in pixels.


    Returns
    -------
    exclusion : list [np.ndarray (Nx, Ny)]
        List of coordinates defining the exclusion.

    """
    if type(coords) != np.ndarray:
        raise TypeError(
            "Coordinate of exclusion of invalid type. Must of `numpy.ndarray` type."
        )
    elif np.shape(coords)[1] != 2:
        raise ValueError(
            "Coordinate input array of incorrect size. Must be of size (n, 2)."
        )
    elif type(radius) != int:
        radius = int(radius)
    elif int(radius) < 0:
        raise ValueError(
            "Radius of circular exclusion invalid. Must be positive integer."
        )
    exclusions = []
    for i in range(np.shape(coords)[0]):
        exclusions.append(
            circular_exclusion(coord=coords[i, :], radius=radius, size=size)
        )
    return exclusions
