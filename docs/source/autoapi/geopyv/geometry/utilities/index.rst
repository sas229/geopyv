:py:mod:`geopyv.geometry.utilities`
===================================

.. py:module:: geopyv.geometry.utilities

.. autoapi-nested-parse::

   Utilities module for geopyv.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   geopyv.geometry.utilities.PolyArea
   geopyv.geometry.utilities.area_to_length
   geopyv.geometry.utilities.plot_triangulation



.. py:function:: PolyArea(pts)

   A function that returns the area of the input polygon.

   :param pts: Clockwise/anti-clockwise ordered coordinates.
   :type pts: `numpy.ndarray` (Nx,2)


.. py:function:: area_to_length(area)

   Function that returns a characteristic length given
   an element area, based on an equilateral triangle.

   :param area: Element area.
   :type area: float

   :returns: **length** -- Characteristic length.
   :rtype: float


.. py:function:: plot_triangulation(elements, x, y)

   Method to compute a first order triangulation from a
   second order element connectivity array and coordinates.

   :param elements: Element connectivity array.
   :type elements: np.ndarray (Nx, 6)
   :param x: Horizontal coordinate array.
   :type x: np.ndarray (Nx, 1)
   :param y: Vertical coordinate array.
   :type y: np.ndarray (Nx, 1)

   :returns: * **mesh_triangulation** (*np.ndarray (Nx, 7)*) -- Mesh triangulation array for plot purposes forming
               closed triangles.
             * **x_p** (*np.ndarray (Nx, 1)*) -- Horizontal coordinate of triangle vertices.
             * **y_p** (*np.ndarray (Nx, 1)*) -- Vertical coordinate of triangle vertices.


