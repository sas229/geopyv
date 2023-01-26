:py:mod:`geopyv.plots`
======================

.. py:module:: geopyv.plots


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   geopyv.plots.inspect_subset
   geopyv.plots.convergence_subset
   geopyv.plots.convergence_mesh
   geopyv.plots.contour_mesh
   geopyv.plots.quiver_mesh
   geopyv.plots.inspect_mesh



.. py:function:: inspect_subset(data, mask, show, block, save)

   Function to show the subset and associated quality metrics.


.. py:function:: convergence_subset(data, show, block, save)

   Function to plot subset convergence.


.. py:function:: convergence_mesh(data, quantity, show, block, save)

   Function to plot subset convergence.


.. py:function:: contour_mesh(data, quantity, imshow, colorbar, ticks, mesh, alpha, levels, axis, xlim, ylim, show, block, save)

   Function to plot contours of mesh data.


.. py:function:: quiver_mesh(data, scale, imshow, mesh, axis, xlim, ylim, show, block, save)

   Function to plot contours of mesh data.


.. py:function:: inspect_mesh(data, show, block, save)

   Function to inspect the mesh.


