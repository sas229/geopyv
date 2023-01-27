:py:mod:`geopyv.mesh`
=====================

.. py:module:: geopyv.mesh


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   geopyv.mesh.MeshBase
   geopyv.mesh.Mesh
   geopyv.mesh.MeshResults




Attributes
~~~~~~~~~~

.. autoapisummary::

   geopyv.mesh.log


.. py:data:: log

   

.. py:class:: MeshBase

   Mesh base class to be used as a mixin.

   .. py:method:: inspect(subset=None, show=True, block=True, save=None)

      Method to show the mesh and associated subset quality metrics.


   .. py:method:: convergence(subset=None, quantity=None, show=True, block=True, save=None)

      Method to plot the rate of convergence for the mesh.


   .. py:method:: contour(quantity='C_ZNCC', imshow=True, colorbar=True, ticks=None, mesh=False, alpha=0.75, levels=None, axis=None, xlim=None, ylim=None, show=True, block=True, save=None)

      Method to plot the contours of a given measure.


   .. py:method:: quiver(scale=1, imshow=True, mesh=False, axis=None, xlim=None, ylim=None, show=True, block=True, save=None)

      Method to plot a quiver plot of the displacements.



.. py:class:: Mesh(*, f_img, g_img, target_nodes=1000, boundary=None, exclusions=[], size_lower_bound=1, size_upper_bound=1000)

   Bases: :py:obj:`MeshBase`

   Initialisation of geopyv mesh object.

   .. py:method:: set_target_nodes(target_nodes)

      Method to create a mesh with a target number of nodes.


   .. py:method:: solve(*, seed_coord=None, template=None, max_iterations=15, max_norm=0.001, adaptive_iterations=0, method='ICGN', order=1, tolerance=0.7, alpha=0.5, beta=2)



.. py:class:: MeshResults(data)

   Bases: :py:obj:`MeshBase`

   Initialisation of geopyv MeshResults class.


