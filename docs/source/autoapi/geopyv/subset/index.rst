:py:mod:`geopyv.subset`
=======================

.. py:module:: geopyv.subset


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   geopyv.subset.SubsetBase
   geopyv.subset.Subset
   geopyv.subset.SubsetResults




Attributes
~~~~~~~~~~

.. autoapisummary::

   geopyv.subset.log


.. py:data:: log

   

.. py:class:: SubsetBase

   Subset base class to be used as a mixin. Contains plot functionality.

   .. py:method:: inspect(show=True, block=True, save=None)

      Method to show the subset and associated quality metrics using :mod:`~geopyv.plots.inspect_subset`.

      :param show: Control whether the plot is displayed.
      :type show: bool, optional
      :param block: Control whether the plot blocks execution until closed.
      :type block: bool, optional
      :param save: Name to use to save plot. Uses default extension of `.png`.
      :type save: str, optional

      :returns: * **fig** (`matplotlib.pyplot.figure`) -- Figure object.
                * **ax** (`matplotlib.pyplot.axes`) -- Axes object.

      .. note::
          * The figure and axes objects can be returned allowing standard matplotlib functionality to be used to augment the plot generated. See the :ref:plots tutorial <`Plots Tutorial>` for guidance.

      .. seealso::
          :meth:`~geopyv.plots.inspect_subset`



   .. py:method:: convergence(show=True, block=True, save=None)

      Method to plot the rate of convergence for the subset using :mod:`~geopyv.plots.convergence_subset`.

      :param show: Control whether the plot is displayed.
      :type show: bool, optional
      :param block: Control whether the plot blocks execution until closed.
      :type block: bool, optional
      :param save: Name to use to save plot. Uses default extension of `.png`.
      :type save: str, optional

      :returns: * **fig** (*matplotlib.pyplot.figure*) -- Figure object.
                * **ax** (`matplotlib.pyplot.axes`) -- Axes object.

      .. note::
          * The figure and axes objects can be returned allowing standard matplotlib functionality to be used to augment the plot generated. See the :ref:plots tutorial <`Plots Tutorial>` for guidance.

      .. warning::
          * Can only be used once the subset has been solved using the :meth:`~geopyv.subset.Subset.solve` method.

      .. seealso::
          :meth:`~geopyv.plots.convergence_subset`




.. py:class:: Subset(*, f_coord=None, f_img=None, g_img=None, template=None)

   Bases: :py:obj:`SubsetBase`

   Initialisation of geopyv subset object.

   :param coord: Subset coordinates.
   :type coord: `numpy.ndarray` (x, y), optional
   :param f_img: Reference image of geopyv.image.Image class, instantiated by :mod:`~geopyv.image.Image`.
   :type f_img: geopyv.image.Image, optional
   :param g_img: Target image of geopyv.imageImage class, instantiated by :mod:`~geopyv.image.Image`.
   :type g_img: geopyv.image.Image, optional
   :param template: Subset template object, instantiated by :mod:`~geopyv.templates.Circle` or :mod:`~geopyv.templates.Square`.
   :type template: geopyv.templates.Template, optional

   .. attribute:: data

      Data object containing all settings and results. See the data structure :ref:`here <subset_data_structure>`.

      :type: dict

   .. attribute:: solved

      Boolean to indicate if the subset has been solved.

      :type: bool

   .. py:method:: solve(*, max_norm=0.001, max_iterations=15, order=1, p_0=None, tolerance=0.7, method='ICGN')

      Method to solve for the subset displacements using the various methods.

      :param max_norm: Exit criterion for norm of increment in warp function. Defaults to value of
                       :math:`1 \cdot 10^{-3}`.
      :type max_norm: float, optional
      :param max_iterations: Exit criterion for number of Gauss-Newton iterations. Defaults to value
                             of 50.
      :type max_iterations: int, optional
      :param order: Warp function order. Options are 1 and 2.
      :type order: int
      :param p_0: 1D array of warp function parameters with `float` type.
      :type p_0: ndarray, optional
      :param tolerance: Correlation coefficient tolerance. Defaults to a value of 0.7.
      :type tolerance: float, optional
      :param method: Solution method. Options are FAGN and ICGN. Default is ICGN since it
                     is faster.
      :type method: str

      :returns: **solved** -- Boolean to indicate if the subset instance has been solved.
      :rtype: `bool`

      .. note::
          * The warp function parameter array can be used to precondition the computation if passed non-zero values.
          * Otherwise, the initial guess at the subset displacement is performed by
            :meth:`~_get_initial_guess`.
          * If not specified, the solver defaults to a first order warp function.
          * For guidance on how to use this class see the subset tutorial :ref:`here <Subset Tutorial>`.


      .. seealso::
          :meth:`~_get_initial_guess_size`
          :meth:`~_get_initial_guess`




.. py:class:: SubsetResults(data)

   Bases: :py:obj:`SubsetBase`

   Subset results object for geopyv.

   :param data: geopyv data dict from Subset object.
   :type data: dict

   .. attribute:: data

      geopyv data dict from Subset object.

      :type: dict

   .. note::
       * Contains all of the plot functionality provied by :class:`~geopyv.subset.SubsetBase` but none of the algorithms provided by :class:`~geopyv.subset.Subset` (i.e. you can't use this to re-analyse images). Purely used to store data and interrogate results.

   .. warning::
       * To re-analyse data instantiate a new object using :class:`~geopyv.subset.Subset` and use the :class:`~geopyv.subset.Subset.solve` method.



