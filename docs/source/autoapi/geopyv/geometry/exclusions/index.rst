:py:mod:`geopyv.geometry.region`
====================================

.. py:module:: geopyv.geometry.region

.. autoapi-nested-parse::

   Region module for geopyv.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   geopyv.geometry.region.Circle
   geopyv.geometry.region.Path
   geopyv.geometry.region.Region
   geopyv.geometry.region.RegionBase
   geopyv.geometry.region.RegionResults

.. py:function:: circular_exclusion(coord, radius, size)

   Function to define an array of circular exclusion coordinates.

   :param coord: Coordinate of exclusion in pixels.
   :type coord: `numpy.ndarray` (x, y)
   :param radius: Radius of exclusion in pixels.
   :type radius: int
   :param size: Required subset spacing on exclusion boundary in pixels.
   :type size: int

   :returns: **exclusion** -- Array of coordinates defining the exclusion.
   :rtype: np.ndarray (Nx,Ny)


.. py:function:: circular_exclusion_list(coords, radius, size)

   Function to define a list of circular exclusion coordinates.


   :param coord: Coordinate of exclusion in pixels.
   :type coord: np.ndarray (x,y)
   :param radius: Radius of exclusion in pixels.
   :type radius: int
   :param size: Required subset spacing on exclusion boundary in pixels.
   :type size: int

   :returns: **exclusion** -- List of coordinates defining the exclusion.
   :rtype: list [np.ndarray (Nx, Ny)]
