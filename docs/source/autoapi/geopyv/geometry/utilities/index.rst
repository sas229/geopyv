:py:mod:`geopyv.geometry.utilities`
===================================

.. py:module:: geopyv.geometry.utilities


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   geopyv.geometry.utilities.area_to_length
   geopyv.geometry.utilities.plot_triangulation



.. py:function:: area_to_length(area)

   Function that returns a characteristic length given an element area, based on an equilateral triangle.

   :param area: Element area.
   :type area: float


.. py:function:: plot_triangulation(elements, x, y)

   Method to compute a first order triangulation from a second order element.

   :param elements: Element connectivity array.
   :type elements: np.ndarray
   :param x: Horizontal coordinate array.
   :type x: np.ndarray
   :param y: Vertical coordinate array.
   :type y: np.ndarray


