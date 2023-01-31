:py:mod:`geopyv.plots`
======================

.. py:module:: geopyv.plots

.. autoapi-nested-parse::

   Plots module for geopyv.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   geopyv.plots.contour_mesh
   geopyv.plots.convergence_mesh
   geopyv.plots.convergence_subset
   geopyv.plots.inspect_mesh
   geopyv.plots.inspect_subset
   geopyv.plots.quiver_mesh



.. py:function:: contour_mesh(data, quantity, imshow, colorbar, ticks, mesh, alpha, levels, axis, xlim, ylim, show, block, save)

   Function to plot contours of mesh data.


.. py:function:: convergence_mesh(data, quantity, show, block, save)

   Function to plot Mesh convergence.

   :param data: Mesh data dict.
   :type data: dict
   :param quantity: Quantity to plot. Options are "C_ZNCC", "iterations", or "norm".
   :type quantity: str
   :param show: Control whether the plot is displayed.
   :type show: bool
   :param block: Control whether the plot blocks execution until closed.
   :type block: bool
   :param save: Name to use to save plot. Uses default extension of `.png`.
   :type save: str

   :returns: * **fig** (*matplotlib.pyplot.figure*) -- Figure object.
             * **ax** (*matplotlib.pyplot.axes*) -- Axes object.

   .. note::
       * The figure and axes objects can be returned allowing standard matplotlib functionality to be used to augment the plot generated. See the :ref:`plots tutorial <Plots Tutorial>` for guidance.

   .. seealso::
       :meth:`~geopyv.mesh.MeshBase.convergence`



.. py:function:: convergence_subset(data, show, block, save)

   Function to plot Subset convergence.

   :param data: Subset data dict.
   :type data: dict
   :param show: Control whether the plot is displayed.
   :type show: bool
   :param block: Control whether the plot blocks execution until closed.
   :type block: bool
   :param save: Name to use to save plot. Uses default extension of `.png`.
   :type save: str

   :returns: * **fig** (*matplotlib.pyplot.figure*) -- Figure object.
             * **ax** (*matplotlib.pyplot.axes*) -- Axes object.

   .. note::
       * The figure and axes objects can be returned allowing standard matplotlib functionality to be used to augment the plot generated. See the :ref:`plots tutorial <Plots Tutorial>` for guidance.

   .. seealso::
       :meth:`~geopyv.subset.SubsetBase.convergence`



.. py:function:: inspect_mesh(data, show, block, save)

   Function to inspect Mesh topology.

   :param data: Mesh data dict.
   :type data: dict
   :param show: Control whether the plot is displayed.
   :type show: bool
   :param block: Control whether the plot blocks execution until closed.
   :type block: bool
   :param save: Name to use to save plot. Uses default extension of `.png`.
   :type save: str

   :returns: * **fig** (*matplotlib.pyplot.figure*) -- Figure object.
             * **ax** (*matplotlib.pyplot.axes*) -- Axes object.

   .. note::
       * The figure and axes objects can be returned allowing standard matplotlib functionality to be used to augment the plot generated. See the :ref:`plots tutorial <Plots Tutorial>` for guidance.

   .. seealso::
       :meth:`~geopyv.mesh.MeshBase.inspect`



.. py:function:: inspect_subset(data, mask, show, block, save)

   Function to show the Subset and associated quality metrics.

   :param data: Subset data dict.
   :type data: dict
   :param mask: Subset mask.
   :type mask: `numpy.ndarray`
   :param show: Control whether the plot is displayed.
   :type show: bool
   :param block: Control whether the plot blocks execution until closed.
   :type block: bool
   :param save: Name to use to save plot. Uses default extension of `.png`.
   :type save: str

   :returns: * **fig** (*matplotlib.pyplot.figure*) -- Figure object.
             * **ax** (*matplotlib.pyplot.axes*) -- Axes object.

   .. note::
       * The figure and axes objects can be returned allowing standard matplotlib functionality to be used to augment the plot generated. See the :ref:`plots tutorial <Plots Tutorial>` for guidance.

   .. seealso::
       :meth:`~geopyv.subset.SubsetBase.inspect`



.. py:function:: quiver_mesh(data, scale, imshow, mesh, axis, xlim, ylim, show, block, save)

   Function to plot quiver plot of mesh data.
