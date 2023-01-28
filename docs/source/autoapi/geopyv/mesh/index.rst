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

   Mesh class for geopyv.

   :param f_img: Reference image of geopyv.image.Image class, instantiated by :mod:`~geopyv.image.Image`.
   :type f_img: geopyv.image.Image, optional
   :param g_img: Target image of geopyv.imageImage class, instantiated by :mod:`~geopyv.image.Image`.
   :type g_img: geopyv.image.Image, optional
   :param target_nodes: Target number of nodes.
   :type target_nodes: int, optional
   :param boundary: Array of coordinates to define the mesh boundary.
   :type boundary: `numpy.ndarray` (Nx,Ny)
   :param exclusions: List of `numpy.ndarray` to define the mesh exclusions.
   :type exclusions: list, optional
   :param size_lower_bound: Lower bound on element size. Defaults to a value of 1.
   :type size_lower_bound: int, optional
   :param upper_lower_bound: Lower bound on element size. Defaults to a value of 1000.
   :type upper_lower_bound: int, optional

   .. attribute:: data

      Data object containing all settings and results. See the data structure :ref:`here <mesh_data_structure>`.

      :type: dict

   .. attribute:: solved

      Boolean to indicate if the mesh has been solved.

      :type: bool

   .. py:method:: set_target_nodes(target_nodes)

      Method to create a mesh with a target number of nodes.

      :param target_nodes: Target number of nodes.
      :type target_nodes: int

      .. note::
          * This method can be used to update the number of target nodes.
          * It will generate a new initial mesh with the specified target number of nodes.



   .. py:method:: solve(*, seed_coord=None, template=None, max_norm=0.001, max_iterations=15, order=1, tolerance=0.7, method='ICGN', adaptive_iterations=0, alpha=0.5, beta=2)

      Method to solve for the mesh.

      :param max_norm: Exit criterion for norm of increment in warp function. Defaults to value of
                       :math:`1 \cdot 10^{-3}`.
      :type max_norm: float, optional
      :param max_iterations: Exit criterion for number of Gauss-Newton iterations. Defaults to value
                             of 50.
      :type max_iterations: int, optional
      :param order: Warp function order. Options are 1 and 2.
      :type order: int
      :param tolerance: Correlation coefficient tolerance. Defaults to a value of 0.7.
      :type tolerance: float, optional
      :param method: Solution method. Options are FAGN and ICGN. Default is ICGN since it is faster.
      :type method: str
      :param adaptive_iterations: Number of mesh adaptivity iterations to perform. Defaults to a value of 0.
      :type adaptive_iterations: int, optional
      :param alpha: Mesh adaptivity control parameter. Defaults to a value of 0.5.
      :type alpha: float, optional
      :param beta: Mesh adaptivity control parameter. Defaults to a value of 2.0.
      :type beta: float, optional

      :returns: **solved** -- Boolean to indicate if the subset instance has been solved.
      :rtype: bool



.. py:class:: MeshResults(data)

   Bases: :py:obj:`MeshBase`

   Initialisation of geopyv MeshResults class.


