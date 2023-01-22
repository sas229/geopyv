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

   Subset base class to be used as a mixin.

   .. py:method:: inspect(show=True, block=True, save=None)

      Method to show the subset and associated quality metrics.


   .. py:method:: convergence(show=True, block=True, save=None)

      Method to plot the rate of convergence for the subset.



.. py:class:: Subset(f_coord=None, f_img=None, g_img=None, template=Circle(50))

   Bases: :py:obj:`SubsetBase`

   Subset class for geopyv.

   :param coord: Subset coordinates.
   :type coord: `numpy.ndarray` (x, y)
   :param f_img: Reference image of geopyv.Image class, instantiated by :mod:`~image.Image`.
   :type f_img: geopyv.Image
   :param g_img: Target image of geopyv.Image class, instantiated by :mod:`~image.Image`.
   :type g_img: geopyv.Image
   :param template: Subset template object.
   :type template: `geopyv.Template`

   .. attribute:: f_img

      Reference image of geopyv.image.Image class, instantiated by :mod:`~image.Image`.

      :type: `geopyv.Image`

   .. attribute:: g_img

      Target image of geopyv.image.Image class, instantiated by :mod:`~image.Image`.

      :type: `geopyv.Image`

   .. attribute:: template

      Subset template object.

      :type: `geopyv.Template`

   .. attribute:: method

      Solver type. Options are 'ICGN' and 'FAGN'.

      :type: `str`

   .. attribute:: init_guess_size

      Size of subset used to define the initial guess, approximated by private method
      :meth:`~_get_initial_guess_size`.

      :type: int

   .. attribute:: f_coord

      1D array of the coordinates of the subset in reference image of type `float`.

      :type: `numpy.ndarray` (x, y)

   .. attribute:: f_coords

      2D array of subset coordinates in reference image of type `float`.

      :type: `numpy.ndarray` (Nx, 2)

   .. attribute:: grad_f

      Gradients of reference image `f`.

      :type: `numpy.ndarray` (Nx, 2)

   .. attribute:: SSSIG

      Sum of the square of the reference subset intensity gradients.

      :type: float

   .. attribute:: sigma_intensity

      Standard deviaition of the reference subset intensities.

      :type: float

   .. attribute:: p_0

      1D array of initial warp function parameters of type `float`, used to precondition
      class method :meth:`~solve`.

      :type: `numpy.ndarray` (Nx, 1)

   .. attribute:: p

      1D array of warp function parameters of type `float`, output by class
      method :meth:`~solve`.

      :type: `numpy.ndarray` (Nx, 1)

   .. attribute:: norm

      Custom norm of the increment in the warp function parameters after
      Gao et al. (2015), computed by private method :meth:`~_get_norm`.

      :type: float

   .. attribute:: C_ZNSSD

      Zero-normalised sum of squared differences coefficient, computed by private
      method :meth:`~_get_correlation`.

      :type: float

   .. attribute:: C_ZNCC

      Zero-normalised cross-correlation coefficient, computed by private method
      :meth:`~_get_correlation`.

      :type: float

   .. attribute:: x

      Initial horizontal coordinate.

      :type: float

   .. attribute:: y

      Initial vertical coordinate.

      :type: float

   .. attribute:: u

      Horizontal displacement.

      :type: float

   .. attribute:: v

      Vertical displacement.

      :type: float

   .. attribute:: x_f

      Final horizontal coordinate.self.initialised

      :type: float

   .. attribute:: y_f

      Final vertical coordinate.

      :type: float

   .. attribute:: settings

      Dictionary of settings.

      :type: dict

   .. attribute:: quality

      Dictionary of image quality measures.

      :type: dict

   .. attribute:: results

      Dictionary of results.

      :type: dict

   .. py:method:: solve(max_norm=0.001, max_iterations=15, p_0=np.zeros(6), tolerance=0.7, method='ICGN')

      Method to solve for the subset displacements using the various methods.

      :param max_norm: Exit criterion for norm of increment in warp function. Defaults to value of
                       :math:`1 \cdot 10^{-5}`.
      :type max_norm: float, optional
      :param max_iterations: Exit criterion for number of Gauss-Newton iterations. Defaults to value
                             of 50.
      :type max_iterations: int, optional
      :param p_0: 1D array of warp function parameters with `float` type.
      :type p_0: ndarray, optional
      :param method: Solution method. Options are FAGN, WFAGN and ICGN. Default is ICGN since it
                     is faster.
      :type method: str

      .. note::
          * If all members of the warp function parameter array are zero, then an
            initial guess at the subset displacement is performed by
            :meth:`~_get_initial_guess`.
          * Otherwise, if any members of the warp function parameter array are
            non-zero, the array is used to precondition the ICGN computation directly.
          * If not specified, the solver defaults to a first order warp function.
          * If an array length of 12 is specified a second order warp function is
            assumed.

      .. seealso::
          :meth:`~_get_initial_guess_size`
          :meth:`~_get_initial_guess`


   .. py:method:: _load_img(message)

      Private method to open a file dialog and slect an image.


   .. py:method:: _load_f_img()

      Private method to load the reference image.


   .. py:method:: _load_g_img()

      Private method to load the target image.


   .. py:method:: _get_initial_guess_size()

      Private method to estimate the size of square subset to use in the
      initial guess.


   .. py:method:: _get_initial_guess()

      Private method to compute an initial guess of the subset displacement using
      OpenCV function :py:meth:`cv2.matchTemplate` and the Normalised
      Cross-Correlation (NCC) criteria.



.. py:class:: SubsetResults(data)

   Bases: :py:obj:`SubsetBase`

   SubsetResults class for geopyv.

   :param data: geopyv data dict from Subset object.
   :type data: dict

   .. attribute:: data

      geopyv data dict from Subset object.

      :type: dict


