import numpy as np

def area_to_length(area):
        """Function that returns a characteristic length given an element area, based on an equilateral triangle.
        
        Parameters
        ----------
        area : float
            Element area.
        """

        return np.sqrt(4*abs(area)/np.sqrt(3))

def plot_triangulation(elements, x, y):
    """Method to compute a first order triangulation from a second order element.

    Parameters
    ----------
    elements : np.ndarray
        Element connectivity array.
    x : np.ndarray
        Horizontal coordinate array.
    y : np.ndarray
        Vertical coordinate array.
    """
    plot_elements = []
    x_p = []
    y_p = []
    for element in elements:
        plot_elements.append([element[0], element[3], element[5]])
        plot_elements.append([element[1], element[3], element[4]])
        plot_elements.append([element[2], element[4], element[5]])
        plot_elements.append([element[3], element[4], element[5]])
        x_p.append([x[element[0]], x[element[3]], x[element[1]], x[element[4]], x[element[2]], x[element[5]], x[element[0]]])
        y_p.append([y[element[0]], y[element[3]], y[element[1]], y[element[4]], y[element[2]], y[element[5]], y[element[0]]])
    mesh_triangulation = np.asarray(plot_elements)
    x_p = np.asarray(x_p)
    y_p = np.asarray(y_p)

    return mesh_triangulation, x_p, y_p