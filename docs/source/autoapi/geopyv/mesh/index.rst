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

      Method to show the mesh and associated quality metrics.


   .. py:method:: convergence(subset=None, show=True, block=True, save=None)

      Method to plot the rate of convergence for the mesh.


   .. py:method:: contour(quantity='C_ZNCC', imshow=True, colorbar=True, ticks=None, mesh=False, alpha=0.75, levels=None, axis=None, xlim=None, ylim=None, show=True, block=True, save=None)

      Method to plot the contours of a given measure.


   .. py:method:: quiver()

      Method to plot a quiver plot of the displacements.



.. py:class:: Mesh(f_img, g_img, target_nodes=1000, boundary=None, exclusions=[], size_lower_bound=1, size_upper_bound=1000)

   Bases: :py:obj:`MeshBase`

   Mesh base class to be used as a mixin.

   .. py:method:: set_target_nodes(target_nodes)

      Method to create a mesh with a target number of nodes.


   .. py:method:: solve(seed_coord=None, template=Circle(50), max_iterations=15, max_norm=0.001, adaptive_iterations=0, method='ICGN', order=1, tolerance=0.7, alpha=0.5, beta=2)


   .. py:method:: _update_mesh()

      Private method to update the mesh variables.


   .. py:method:: _find_seed_node()

      Private method to find seed node given seed coordinate.


   .. py:method:: _define_RoI()

      Private method to define the RoI.


   .. py:method:: _initial_mesh()

      Private method to optimize the element size to generate approximately the desired number of elements.


   .. py:method:: _adaptive_mesh()


   .. py:method:: _uniform_remesh(size, boundary, segments, curves, target_nodes, size_lower_bound)
      :staticmethod:

      Private method to prepare the initial mesh.


   .. py:method:: _adaptive_remesh(scale, target, nodes, triangulation, areas)
      :staticmethod:


   .. py:method:: _adaptive_subset()


   .. py:method:: _update_subset_bgf()


   .. py:method:: _element_area()

      A private method to calculate the element areas.


   .. py:method:: _element_strains()

      A private method to calculate the elemental strain the "B" matrix relating
      element node displacements to elemental strain.


   .. py:method:: _reliability_guided()

      A private method to perform reliability-guided (RG) PIV analysis.


   .. py:method:: _connectivity(idx)

      A private method that returns the indices of nodes connected to the index node according to the input array.

      :param idx: Index of node.
      :type idx: int
      :param arr: Mesh array.
      :type arr: numpy.ndarray (N)


   .. py:method:: _neighbours(cur_idx, p_0)

      Method to calculate the correlation coefficients and warp functions of the neighbouring nodes.

      :param p_0: Preconditioning warp function.
      :type p_0: numpy.ndarray (N)


   .. py:method:: _store_variables(idx, seed=False)

      Store variables.



.. py:class:: MeshResults(data)

   Bases: :py:obj:`MeshBase`

   MeshResults class for geopyv.

   :param data: geopyv data dict from Mesh object.
   :type data: dict

   .. attribute:: data

      geopyv data dict from Mesh object.

      :type: dict


