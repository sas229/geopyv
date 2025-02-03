:py:mod:`geopyv.field`
======================

.. py:module:: geopyv.field

.. autoapi-nested-parse::

   Field module for geopyv.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   geopyv.field.Field
   geopyv.field.FieldBase
   geopyv.field.FieldResults


.. py:class:: Field(*, series=None, target_particles=1000, track=True, boundary=None, exclusions=[], coordinates=None, volumes=None, stresses=np.zeros(6), strains=np.zeros(6), depth = 1000, ID="",)

   Bases: :py:obj:`FieldBase`

   Initialisation of geopyv field object.

   :param series: The subject series object instantiated by :mod:`~geopyv.mesh.Mesh` or :mod:`~geopyv.sequence.Sequence` for field object interpolation.
   :type series: geopyv.mesh.Mesh, geopyv.sequence.Sequence
   :param target_particles: The target number of particles. Defaults to a value of 1000.
   :type target_particles: int, optional
   :param track: Boolean flag to specify if the particles should move (`True`, Lagrangian) 
                  or remain static (`False`, Eulerian). Defaults to a value of True.
   :type track: bool, optional
   :param boundary: The geometry of the region of interest for analysis. May 
                  be a :mod:`~geopyv.geometry.region.Circle` object, 
                  :mod:`~geopyv.geometry.region.Path` object, 
                  `numpy.ndarray` array of coordinates or `None` (uses the 
                  boundary of the series). Defaults to a value of `None`.
   :type boundary: gp.geometry.region.Circle, gp.geometry.region.Path, numpy.ndarray, optional
   :param exclusions: The geometry of the regions to exclude from analysis 
                  within the boundary. May be a `list` of 
                  :mod:`~geopyv.geometry.region.Circle` objects,
                  :mod:`~geopyv.geometry.region.Path` objects or
                  `numpy.ndarray` arrays of coordinates. Defaults to an empty list, [].
   :type exclusions: list, optional
   :param coordinates: Array of coordinates to specify particle positions. Specification 
                  overrides auto-distribution according to the boundary and any exclusions.
                  Defaults to a value of `None`.`
   :type coordinates: numpy.ndarray(N), optional
   :param volumes: Array of volumes to specify initial particle volumes. If coordinates are 
                  specified, particle volumes will default to `1.0` in absence of user-specification.
                  If user-specified, the array must match the shape of the coordinates array. 
   :type volumes: numpy.ndarray(N), optional
   :param stresses: Array of stresses to specify initial stress distribution. If coordinates are
                  specified, then `numpy.ndarray(N,6)` provides the initial individual particle stress states.
                  Instead, if particles are auto-distributed, `numpy.ndarray(2,6)` creates a linearly varying
                  stress distribution with depth (:math:`1^{st}` and :math:`2^{nd}` high and low respectively)
                  according to the series boundary definition. Defaults to a value of `numpy.zeros(6)`. 
                  Alternatively, `numpy.ndarray(6)` creates a uniform stress distribution. Note, Voigt
                  notation is adopted. Defaults to `numpy.zeros(6)`.
   :type stresses: numpy.ndarray(N,6), numpy.ndarray(2,6), numpy.ndarray(6), optional
   :param strains: Array of strains to specify intial accumulated strains. The `numpy.ndarray(N,6)` specifies
                  the previously accumulated/initial strains for the particles individually. Note, Voigt
                  notation is adopted. Defaults to `numpy.zeros(6)` for each particle (i.e. no strain history). 
   :type strains: numpy.ndarray(N,6), optional
   :param depth: The depth of the plane-strain problem (in `px` or `mm` dependent on the series object calibration
                  state). Defaults to a value of 1000 (i.e. :math:`1~m` unit depth).
   :type depth: float, optional
   :param ID: Identification. 
   :type ID: str, optional


.. py:class:: FieldBase

   Bases: :py:obj:`geopyv.object.Object`

   Base class object initialiser.

   :param object_type: Object type.
   :type object_type: str

   .. py:method:: inspect(mesh=True, show=True, block=True, save=None)

      Method to show the particles and associated representative areas.

      :param mesh: Control whether the mesh is plotted. Defaults to True. 
      :type mesh: bool, optional
      :param show: Control whether the plot is displayed.
      :type show: bool, optional
      :param block: Control whether the plot blocks execution until closed.
      :type block: bool, optional
      :param save: Name to use to save plot. Uses default extension of `.png`.
      :type save: str, optional


   .. py:method:: trace(*, quantity="warps", particle_index=None, component=0, imshow=True, colorbar=True, ticks=None, alpha=0.75, axis=True, xlim=None, ylim=None, show=True, block=True, save=None)

      Method to plot an incremental quantity along the particle position path.

      :param quantity: Specifier for which metric to plot along the particle path. May be
                     "warps", "coordinates", "volumes", "stresses". Defaults to a value of "warps".
      :type quantity: str, optional
      :param particle_index: The index of the particle to plot. If unspecified, all particles are plotted.
                     Defaults to a value of `None`.
      :type particle_index: int, optional
      :param component: Specifier for which component of the metric to plot along the particle path.
                     Defaults to a value of 0.
      :type component: int, optional
      :param imshow: Boolean control for whether the reference image is plotted. Defaults to a value of `True`.
      :type imshow: bool, optional
      :param colorbar: Boolean control for whether the colour bar is plotted. Defaults to a value of `True`.
      :type colorbar: bool, optional
      :param ticks: Array to overwrite the default colour bar ticks. Defaults to a value of `None`.
      :type ticks: list, numpy.ndarray, optional
      :param alpha: Control for line opacity. Must be between 0.0-1.0. Defaults to a value of `0.75`.
      :type alpha: float, optional
      :param axis: Control whether the axes are plotted. Defaults to a value of `True`.
      :type axis: bool, optional
      :param xlim: Set the plot x-limits (`lower limit`, `upper limit`). Defaults to a value of `None`.
      :type xlim: array-like, optional
      :param ylim: Set the plot y-limits (`lower limit`, `upper limit`). Defaults to a value of `None`.
      :type ylim: array-like, optional
      :type show: bool, optional
      :param block: Control whether the plot blocks execution until closed.
      :type block: bool, optional
      :param save: Name to use to save plot. Uses default extension of `.png`.
      :type save: str, optional

      :returns: * **fig** (*matplotlib.pyplot.figure*) -- Figure object.
                * **ax** (`matplotlib.pyplot.axes`) -- Axes object.
      
      .. note::
          * The figure and axes objects can be returned allowing standard
            matplotlib functionality to be used to augment the plot generated.
            See the :ref:`plots tutorial <Plots Tutorial>` for guidance.

      .. warning::
          * Can only be used once the field has been solved using the
            :meth:`~geopyv.field.Field.solve` method.

      .. seealso::
          :meth:`~geopyv.plots.trace_particle`

   .. py:method:: contour(*, quantity = "u", window = None, series = None, original = False, exclusions = [], absolute = False, colorbar = True, scale = "lin", ticks=None, alpha=0.75, extend = None,  levels=None, axis=True, xlim=None, ylim=None, show=True, block=True, save=None)

      Method to generate contour plots.

      :param quantity: Specifier for which metric to plot. May be "u" (horizontal displacement), 
                  "v" (vertical displacement), "R" (resultant displacement), "ep_xy" (shear strain),
                  "ep_vol" (volumetric strain), "p" (mean effective stress), "q" (deviatoric stress),
                  "work" or "power". Defaults to a value of "u".
      :type quantity: str, optional
      :param window: Specifier for reference and target indices. e.g. if (0,1), the contour plot is for the 
                     first time increment whereas if (0,-1) the contour plot is for all increments. This allows for
                     accumulation as well as incremental contour plots. Defaults to a value of None (corresponding to 
                     (-2,-1) i.e. the final increment).
      :type window: tuple, optional
      :param series: The subject series object of the field object (used internally for background images). Defaults to a 
                     value of None. 
      :type series: :mod:`~geopyv.sequence.Sequence`, :mod:`~geopyv.mesh.Mesh`, optional
      :param original: Specifier for plotting the reference image. Defaults to a value of `False`.
      :type original: bool, optional
      :param exclusions: Particle indices defining contour plot exclusions. Defaults to a value of [].
      :type exclusions: list, optional
      :param absolute: Specifier for absolutising the increments. Defaults to a value of `False`.
      :type absolute: bool, optional
      :param colorbar: Boolean control for whether the colour bar is plotted. Defaults to a value of `True`.
      :type colorbar: bool, optional
      :param scale: Specifier for colour bar scale. May be "lin" or "log". Defaults to a value of "lin". 
      :type scale: str, optional
      :param ticks: Array to overwrite the default colour bar ticks. Defaults to a value of `None`.
      :type ticks: list, numpy.ndarray, optional
      :param alpha: Control for line opacity. Must be between 0.0-1.0. Defaults to a value of `0.75`.
      :type alpha: float, optional
      :param extend: Specifier of colour bar extensions. May be "both", "max", "min" or "neither". Defaults to
                     a value of `None` (automated specification)
      :type extend: str, optional
      :param levels: Specifier of contour levels. Defaults to a value of `None`.
      :type levels: array-like, optional
      :param axis: Control whether the axes are plotted. Defaults to a value of `True`.
      :type axis: bool, optional
      :param xlim: Set the plot x-limits (`lower limit`, `upper limit`). Defaults to a value of `None`.
      :type xlim: array-like, optional
      :param ylim: Set the plot y-limits (`lower limit`, `upper limit`). Defaults to a value of `None`.
      :type ylim: array-like, optional
      :type show: bool, optional
      :param block: Control whether the plot blocks execution until closed.
      :type block: bool, optional
      :param save: Name to use to save plot. Uses default extension of `.png`.
      :type save: str, optional
      
      :returns: * **fig** (*matplotlib.pyplot.figure*) -- Figure object.
                * **ax** (`matplotlib.pyplot.axes`) -- Axes object.

      .. note::
          * The figure and axes objects can be returned allowing standard
            matplotlib functionality to be used to augment the plot generated.
            See the :ref:`plots tutorial <Plots Tutorial>` for guidance.

      .. warning::
          * Can only be used once the field has been solved using the
            :meth:`~geopyv.field.Field.solve` method.

      .. seealso::
          :meth:`~geopyv.plots.contour_field`

   .. py:method:: history(*, particle_index = 0, quantity="warps", components=None, xlim=None, ylim=None, show=True, block=True, save=None)

      Method to plot particle time history.

      :param particle_index: The index of the particle to plot. If unspecified, all particles are plotted.
                     Defaults to a value of `None`.
      :type particle_index: int, optional
      :param quantity: Specifier for which metric to plot. May be
                     "warps", "coordinates", "volumes", "stresses" or "works". 
                     Defaults to a value of "warps".
      :type quantity: str, optional



      :returns: * **fig** (*matplotlib.pyplot.figure*) -- Figure object.
                * **ax** (`matplotlib.pyplot.axes`) -- Axes object.

      

