"""

Subset module for geopyv.

"""
import logging
import cv2
import os
import numpy as np
import geopyv as gp

log = logging.getLogger(__name__)


class SubsetBase:
    """

    Subset base class to be used as a mixin. Contains plot functionality.

    """

    def inspect(self, show=True, block=True, save=None):
        """

        Method to show the subset and associated quality metrics using
        :mod:`~geopyv.plots.inspect_subset`.

        Parameters
        ----------
        show : bool, optional
            Control whether the plot is displayed.
        block : bool, optional
            Control whether the plot blocks execution until closed.
        save : str, optional
            Name to use to save plot. Uses default extension of `.png`.


        Returns
        -------
        fig :  `matplotlib.pyplot.figure`
            Figure object.
        ax : `matplotlib.pyplot.axes`
            Axes object.


        .. note::
            * The figure and axes objects can be returned allowing standard
              matplotlib functionality to be used to augment the plot generated.
              See the :ref:`plots tutorial <Plots Tutorial>` for guidance.

        .. seealso::
            :meth:`~geopyv.plots.inspect_subset`

        """
        fig, ax = gp.plots.inspect_subset(
            data=self.data, mask=None, show=show, block=block, save=save
        )
        return fig, ax

    def convergence(self, show=True, block=True, save=None):
        """

        Method to plot the rate of convergence for the subset using
        :mod:`~geopyv.plots.convergence_subset`.

        Parameters
        ----------
        show : bool, optional
            Control whether the plot is displayed.
        block : bool, optional
            Control whether the plot blocks execution until closed.
        save : str, optional
            Name to use to save plot. Uses default extension of `.png`.


        Returns
        -------
        fig :  matplotlib.pyplot.figure
            Figure object.
        ax : `matplotlib.pyplot.axes`
            Axes object.


        .. note::
            * The figure and axes objects can be returned allowing standard
              matplotlib functionality to be used to augment the plot generated.
              See the :ref:`plots tutorial <Plots Tutorial>` for guidance.

        .. warning::
            * Can only be used once the subset has been solved using the
              :meth:`~geopyv.subset.Subset.solve` method.

        .. seealso::
            :meth:`~geopyv.plots.convergence_subset`

        """
        if "results" in self.data:
            fig, ax = gp.plots.convergence_subset(
                data=self.data, show=show, block=block, save=save
            )
            return fig, ax
        else:
            log.error("Subset not yet solved. Run the solve() method.")


class Subset(SubsetBase):
    """

    Subset class for geopyv.

    Private Attributes
    ------------------
    _f_img : `geopyv.image.Image`
        Reference image of geopyv.image.Image class, instantiated by
        :mod:`~image.Image`.
    _g_img : `geopyv.image.Image`
        Target image of geopyv.image.Image class, instantiated by
        :mod:`~image.Image`.
    _template: `geopyv.templates.Template`
        Subset template object.
    _method : `str`
        Solver type. Options are 'ICGN' and 'FAGN'.
    _init_guess_size : int
        Size of subset used to define the initial guess, approximated
        by private method
        :meth:`~_get_initial_guess_size`.
    _f_coord : `numpy.ndarray` (x, y)
        1D array of the coordinates of the subset in reference image
        of type `float`.
    _f_coords : `numpy.ndarray` (Nx, 2)
        2D array of subset coordinates in reference image of type `float`.
    _grad_f : `numpy.ndarray` (Nx, 2)
        Gradients of reference image `f`.
    _SSSIG : float
        Sum of the square of the reference subset intensity gradients.
    _sigma_intensity : float
        Standard deviaition of the reference subset intensities.
    _p_0 : `numpy.ndarray` (Nx, 1)
        1D array of initial warp function parameters of type `float`,
        used to precondition class method :meth:`~solve`.
    _p : `numpy.ndarray` (Nx, 1)
        1D array of warp function parameters of type `float`, output by class
        method :meth:`~solve`.
    _norm : float
        Custom norm of the increment in the warp function parameters after
        Gao et al. (2015), computed by private method :meth:`~_get_norm`.
    _C_ZNSSD : float
        Zero-normalised sum of squared differences coefficient, computed by
        private method :meth:`~_get_correlation`.
    _C_ZNCC : float
        Zero-normalised cross-correlation coefficient, computed by private
        method :meth:`~_get_correlation`.
    _x : float
        Initial horizontal coordinate.
    _y : float
        Initial vertical coordinate.
    _u : float
        Horizontal displacement.
    _v : float
        Vertical displacement.
    _x_f : float
        Final horizontal coordinate.
    _y_f : float
        Final vertical coordinate.
    _settings: dict
        Dictionary of settings.
    _quality: dict
        Dictionary of image quality measures.
    _results: dict
        Dictionary of results.

    """

    def __init__(self, *, f_coord=None, f_img=None, g_img=None, template=None):
        """

        Initialisation of geopyv subset object.

        Parameters
        ----------
        coord : `numpy.ndarray` (x, y), optional
            Subset coordinates.
        f_img : geopyv.image.Image, optional
            Reference image of geopyv.image.Image class, instantiated by
            :mod:`~geopyv.image.Image`.
        g_img : geopyv.image.Image, optional
            Target image of geopyv.imageImage class, instantiated by
            :mod:`~geopyv.image.Image`.
        template : geopyv.templates.Template, optional
            Subset template object, instantiated by
            :mod:`~geopyv.templates.Circle` or :mod:`~geopyv.templates.Square`.


        Attributes
        ----------
        data : dict
            Data object containing all settings and results. See the data
            structure :ref:`here <subset_data_structure>`.
        solved : bool
            Boolean to indicate if the subset has been solved.

        """
        log.debug("Initialising geopyv Subset object.")
        self._initialised = False
        self._f_coord = f_coord
        self._f_img = f_img
        self._g_img = g_img
        self._template = template

        # Check types.
        if type(self._f_img) != gp.image.Image:
            self._f_img = self._load_f_img()
        if type(self._g_img) != gp.image.Image:
            self._g_img = self._load_g_img()
        if type(self._f_coord) != np.ndarray:
            self._f_coord = np.empty(2)
            coordinate = gp.gui.selectors.coordinate.CoordinateSelector()
            self._f_coord = coordinate.select(self._f_img, self._template)
        elif np.shape(self._f_coord) != np.shape(np.empty(2)):
            log.error("Template coordinate of invalid shape.")
            coordinate = gp.gui.selectors.coordinate.CoordinateSelector()
            self._f_coord = coordinate.select(self._f_img, self._template)
        if self._template is None:
            self._template = gp.templates.Circle(50)
        elif (
            type(self._template) != gp.templates.Circle
            and type(self._template) != gp.templates.Square
        ):
            log.error(
                "Template not defined in geopyv.templates. "
                "Using default: gp.templates.Circle(50)."
            )
            self._template = gp.templates.Circle(50)

        # Check subset is entirely within the reference image.
        self._x = self._f_coord[0]
        self._y = self._f_coord[1]
        subset_list = np.array(
            [
                [self._x - self._template.size, self._y - self._template.size],
                [self._x + self._template.size, self._y + self._template.size],
                [self._x - self._template.size, self._y + self._template.size],
                [self._x + self._template.size, self._y - self._template.size],
            ]
        )
        if np.any(subset_list < 0):
            raise ValueError(
                "Subset reference partially falls outside reference image."
            )

        # Initialise subset.
        self._get_initial_guess_size()
        output = gp._subset_extensions._init_reference(
            self._f_coord, self._template.coords, self._f_img.QCQT
        )
        self._f_coords = output[0]
        self._f = output[1]
        self._f_m = output[2][0][0]
        self._Delta_f = output[2][1][0]
        self._grad_f = output[3]
        self._SSSIG = output[4][0][0]
        self._sigma_intensity = output[4][1][0]
        self.solved = False
        self._unsolvable = False
        self._initialised = True
        self._quality = {
            "SSSIG": self._SSSIG,
            "sigma_intensity": self._sigma_intensity,
        }

        # Data.
        self.data = {
            "type": "Subset",
            "solved": self.solved,
            "unsolvable": self._unsolvable,
            "images": {
                "f_img": self._f_img.filepath,
                "g_img": self._g_img.filepath,
            },
            "position": {
                "x": self._x,
                "y": self._y,
            },
            "quality": self._quality,
            "template": {
                "shape": self._template.shape,
                "dimension": self._template.dimension,
                "size": self._template.size,
                "n_px": self._template.n_px,
            },
        }
        log.debug("Initialised geopyv Subset object.")

    def solve(
        self,
        *,
        max_norm=1e-3,
        max_iterations=15,
        order=1,
        p_0=None,
        tolerance=0.7,
        method="ICGN"
    ):
        r"""

        Method to solve for the subset displacements using the various methods.

        Parameters
        ----------
        max_norm : float, optional
            Exit criterion for norm of increment in warp function. Defaults to
            value of :math:`1 \cdot 10^{-3}`.
        max_iterations : int, optional
            Exit criterion for number of Gauss-Newton iterations. Defaults to
            value of 50.
        order : int
            Warp function order. Options are 1 and 2.
        p_0 : ndarray, optional
            1D array of warp function parameters with `float` type.
        tolerance: float, optional
            Correlation coefficient tolerance. Defaults to a value of 0.7.
        method : str
            Solution method. Options are FAGN and ICGN. Default is ICGN since
            it is faster.


        Returns
        -------
        solved : `bool`
            Boolean to indicate if the subset instance has been solved.


        .. note::
            * The warp function parameter array can be used to precondition
              the computation if passed non-zero values.
            * Otherwise, the initial guess at the subset displacement is
              performed by :meth:`~_get_initial_guess`.
            * If not specified, the solver defaults to a first order warp
              function.
            * For guidance on how to use this class see the
              :ref:`subset tutorial <Subset Tutorial>`.


        .. seealso::
            :meth:`~_get_initial_guess_size`
            :meth:`~_get_initial_guess`

        """
        # Check other control parameters.
        log.debug("Solving geopyv Subset object.")
        if max_norm < 1e-10:
            log.error("Maximum norm value too small. Suggested default is 1e-3.")
            return False
        elif type(max_iterations) != int:
            log.error(
                "Maximum number of iterations of invalid type. "
                "Must be positive integer."
            )
            return False
        elif max_iterations <= 0:
            log.error(
                "Invalid maximum number of iterations specified. "
                "Must be positive integer."
            )
            return False
        elif type(tolerance) != float:
            log.error(
                "Tolerance of invalid type. Must be float greater than "
                "zero and less than one."
            )
            return False
        elif tolerance < 0:
            log.error(
                "Tolerance must be greater than or equal to zero. "
                "Suggested default is 0.75."
            )
            return False
        elif tolerance > 1:
            log.error("Tolerance must be less than one. Suggested default is 0.75.")
            return False

        # Check warp function order and type.
        if isinstance(p_0, type(None)):
            if order == 1:
                p_0 = np.zeros(6)
            elif order == 2:
                p_0 = np.zeros(12)
            else:
                log.error("Invalid warp function order.")
                return False
        else:
            if type(p_0) != np.ndarray:
                log.error("Warp function of incorrect type.")
                return False
            if order == 1 and np.shape(p_0)[0] != 6:
                log.error("Warp function preconditioning vector of incorrect shape.")
                return False
            if order == 2 and np.shape(p_0)[0] != 12:
                log.error("Warp function preconditioning vector of incorrect shape.")
                return False

        # Store settings.
        self._method = method
        self._order = order
        self._max_norm = max_norm
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._settings = {
            "method": self._method,
            "order": self._order,
            "max_norm": self._max_norm,
            "max_iterations": self._max_iterations,
            "tolerance": self._tolerance,
        }
        self.data.update({"settings": self._settings})

        # Compute initial guess if u and v in deformation parameter vector
        # initialised with zeros, otherwise precondition.
        if p_0[0] == 0 and p_0[1] == 0:
            self._p_init = p_0
            self._get_initial_guess()
            self._p = self._p_init
        else:
            self._p = p_0

        # Call appropriate iterative solver:
        try:
            if self._method == "ICGN" and np.mod(self._p.shape[0], 2) == 0:
                results = gp._subset_extensions._solve_ICGN(
                    self._f_coord,
                    self._f_coords,
                    self._f,
                    self._f_m,
                    self._Delta_f,
                    self._grad_f,
                    self._f_img.QCQT,
                    self._g_img.QCQT,
                    self._p,
                    self._max_norm,
                    self._max_iterations,
                )
            elif self._method == "FAGN" and np.mod(self._p.shape[0], 2) == 0:
                results = gp._subset_extensions._solve_FAGN(
                    self._f_coord,
                    self._f_coords,
                    self._f,
                    self._f_m,
                    self._Delta_f,
                    self._grad_f,
                    self._f_img.QCQT,
                    self._g_img.QCQT,
                    self._p,
                    self._max_norm,
                    self._max_iterations,
                )
            # Unpack results.
            self._g_coords = results[0]
            self._g = results[1]
            self._g_m = results[2][0][0]
            self._g_m = results[2][1][0]
            self._iterations = np.max(results[3][0, :]).astype(int)
            self._history = results[3][:, : self._iterations]
            self._norm = self._history[1][-1]
            self._C_ZNCC = self._history[2][-1]
            self._C_ZNSSD = self._history[3][-1]
            self._p = results[4]
            self._u = self._p[0][0]
            self._v = self._p[1][0]
            self._x_f = self._x + self._u
            self._y_f = self._y + self._v

            # Check for tolerance.
            if self._C_ZNCC > self._tolerance:
                self.solved = True
                log.debug("Subset solved.")
                log.debug(
                    "Initial horizontal coordinate: {x_i} (px); "
                    "Initial vertical coordinate: {y_i} (px)".format(
                        x_i=self._x, y_i=self._y
                    )
                )
                log.debug(
                    "Horizontal displacement: {u} (px); "
                    "Vertical displacement: {v} (px)".format(u=self._u, v=self._v)
                )
                log.debug(
                    "Correlation coefficient: "
                    "CC = {C_ZNCC} (-), SSD = {C_ZNSSD} (-)".format(
                        C_ZNCC=self._C_ZNCC, C_ZNSSD=self._C_ZNSSD
                    )
                )
                log.debug(
                    "Final horizontal coordinate: {x_f} (px); "
                    "Final vertical coordinate: {y_f} (px)".format(
                        x_f=self._x_f, y_f=self._y_f
                    )
                )

            # Pack results.
            self.data["solved"] = self.solved
            self.data["unsolvable"] = self._unsolvable
            self._results = {
                "u": self._u,
                "v": self._v,
                "p": self._p,
                "history": self._history,
                "iterations": self._iterations,
                "norm": self._norm,
                "C_ZNCC": self._C_ZNCC,
                "C_ZNSSD": self._C_ZNSSD,
            }
            self.data.update({"results": self._results})

            # Return solved boolean.
            log.debug("Solved geopyv Subset object.")
            return self.solved

        except Exception:
            log.error("Subset not solved.")
            return False

    def _load_img(self, message):
        """

        Private method to open a file dialog and select an image.

        """
        directory = os.getcwd()
        dialog = gp.gui.selectors.image.ImageSelector()
        imgpath = dialog.get_path(directory, message)
        img = gp.image.Image(imgpath)
        return img

    def _load_f_img(self):
        """

        Private method to load the reference image.

        """
        log.warn("No reference image supplied. Please select the reference image.")
        return self._load_img("Select reference image.")

    def _load_g_img(self):
        """

        Private method to load the target image.

        """
        log.warn("No target image supplied. Please select the target image.")
        return self._load_img("Select target image.")

    def _get_initial_guess_size(self):
        """

        Private method to estimate the size of square subset to use in the
        initial guess.

        """
        self._initial_guess_size = np.round(np.sqrt(self._template.n_px), 1)

    def _get_initial_guess(self):
        """

        Private method to compute an initial guess of the subset displacement
        using OpenCV function :py:meth:`cv2.matchTemplate` and the Normalised
        Cross-Correlation (NCC) criteria.

        """
        # Extract square subset for initial guess.
        x = self._f_coord.item(0)
        y = self._f_coord.item(1)
        x_min = (np.round(x, 0) - self._initial_guess_size / 2).astype(int)
        x_max = (np.round(x, 0) + self._initial_guess_size / 2).astype(int)
        y_min = (np.round(y, 0) - self._initial_guess_size / 2).astype(int)
        y_max = (np.round(y, 0) + self._initial_guess_size / 2).astype(int)
        subset = self._f_img.image_gs.astype(np.float32)[y_min:y_max, x_min:x_max]

        # Apply template matching technique.
        res = cv2.matchTemplate(
            self._g_img.image_gs.astype(np.float32), subset, cv2.TM_CCORR_NORMED
        )
        max_loc = cv2.minMaxLoc(res)[3]

        # Initialised warp vector with preconditioned affine displacements.
        self._p_init[0] = (max_loc[0] + self._initial_guess_size / 2) - x
        self._p_init[1] = (max_loc[1] + self._initial_guess_size / 2) - y


class SubsetResults(SubsetBase):
    def __init__(self, data):
        """

        Subset results object for geopyv.

        Parameters
        ----------
        data : dict
            geopyv data dict from Subset object.


        Attributes
        ----------
        data : dict
            geopyv data dict from Subset object.


        .. note::
            * Contains all of the plot functionality provied by
              :class:`~geopyv.subset.SubsetBase` but none of the algorithms
              provided by :class:`~geopyv.subset.Subset` (i.e. you can't use
              this to re-analyse images). Purely used to store data and
              interrogate results.

        .. warning::
            * To re-analyse data instantiate a new object using
              :class:`~geopyv.subset.Subset` and use the
              :class:`~geopyv.subset.Subset.solve` method.

        """
        self.data = data
