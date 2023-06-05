"""

Subset module for geopyv.

"""
import logging
import cv2
import numpy as np
import geopyv as gp
from geopyv.object import Object
import traceback

log = logging.getLogger(__name__)


class SubsetBase(Object):
    """

    Subset base class to be used as a mixin. Contains plot functionality.

    """

    def __init__(self):
        super().__init__(object_type="Subset")
        """

        Subset base class initialiser.

        """

    def inspect(self, warp=False, show=True, block=True, save=None):
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

        # Check input.
        self._report(gp.check._check_type(warp, "warp", [bool]), "TypeError")
        if warp:
            # Check if solved.
            if self.data["solved"] is not True:
                log.error(
                    "Subset not yet solved therefore no warp data to plot. "
                    "First, run :meth:`~geopyv.subset.Subset.solve()` to solve."
                )
                raise ValueError(
                    "Mesh not yet solved therefore no warp data to plot. "
                    "First, run :meth:`~geopyv.subset.Subset.solve()` to solve."
                )
        self._report(gp.check._check_type(show, "show", [bool]), "TypeError")
        self._report(gp.check._check_type(block, "block", [bool]), "TypeError")
        self._report(gp.check._check_type(save, "save", [str, type(None)]), "TypeError")

        # Inspect subset.
        if warp:
            fig, ax = gp.plots.inspect_subset_warp(
                data=self.data,
                mask=None,
                show=show,
                block=block,
                save=save,
            )
        else:
            fig, ax = gp.plots.inspect_subset(
                data=self.data,
                mask=None,
                show=show,
                block=block,
                save=save,
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

        # Check input.
        self._report(gp.check._check_type(show, "show", [bool]), "TypeError")
        self._report(gp.check._check_type(block, "block", [bool]), "TypeError")
        self._report(gp.check._check_type(save, "save", [str, type(None)]), "TypeError")

        # Plot convergence.
        if "results" in self.data:
            fig, ax = gp.plots.convergence_subset(
                data=self.data, show=show, block=block, save=save
            )
            return fig, ax
        else:
            log.error("Subset not yet solved. Run the solve() method.")

    def _report(self, msg, error_type):
        if msg and error_type != "Warning":
            log.error(msg)
        elif msg and error_type == "Warning":
            log.warning(msg)
            return True
        if error_type == "ValueError" and msg:
            raise ValueError(msg)
        elif error_type == "TypeError" and msg:
            raise TypeError(msg)
        elif error_type == "IndexError" and msg:
            raise IndexError(msg)


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

        # Check types.
        if self._report(
            gp.check._check_type(f_img, "f_img", [gp.image.Image]), "Warning"
        ):
            f_img = gp.io._load_f_img()
        if self._report(
            gp.check._check_type(g_img, "g_img", [gp.image.Image]), "Warning"
        ):
            g_img = gp.io._load_g_img()
        if template is None:
            template = gp.templates.Circle(50)
        types = [gp.templates.Circle, gp.templates.Square]
        self._report(gp.check._check_type(template, "template", types), "TypeError")
        if self._report(
            gp.check._check_type(f_coord, "f_coord", [np.ndarray]), "Warning"
        ):
            selector = gp.gui.selectors.coordinate.CoordinateSelector()
            f_coord = selector.select(f_img, template)
        elif self._report(gp.check._check_dim(f_coord, "f_coord", 1), "Warning"):
            selector = gp.gui.selectors.coordinate.CoordinateSelector()
            f_coord = selector.select(f_img, template)
        elif self._report(gp.check._check_axis(f_coord, "f_coord", 0, [2]), "Warning"):
            selector = gp.gui.selectors.coordinate.CoordinateSelector()
            f_coord = selector.select(f_img, template)

        # Store.
        self._initialised = False
        self._f_coord = f_coord
        self._f_img = f_img
        self._g_img = g_img
        self._template = template

        # Check subset is entirely within the reference image.
        self._x = self._f_coord[0]
        self._y = self._f_coord[1]
        subset_list = np.array(
            [
                [
                    self._x - self._template.size,
                    self._y - self._template.size,
                ],
                [
                    self._x + self._template.size,
                    self._y + self._template.size,
                ],
                [
                    self._x - self._template.size,
                    self._y + self._template.size,
                ],
                [
                    self._x + self._template.size,
                    self._y - self._template.size,
                ],
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
        max_norm=1e-5,
        max_iterations=50,
        order=1,
        warp_0=np.zeros(6),
        tolerance=0.75,
        method="ICGN",
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
        warp_0 : ndarray, optional
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
        check = gp.check._check_type(max_norm, "max_norm", [float])
        if check:
            try:
                max_norm = float(max_norm)
                self._report(
                    gp.check._conversion(max_norm, "max_norm", float), "Warning"
                )
            except Exception:
                self._report(check, "TypeError")
        self._report(gp.check._check_range(max_norm, "max_norm", 1e-20), "ValueError")
        check = gp.check._check_type(max_iterations, "max_iterations", [int])
        if check:
            try:
                max_iterations = int(max_iterations)
                self._report(
                    gp.check._conversion(max_iterations, "max_iterations", int),
                    "Warning",
                )
            except Exception:
                self._report(check, "TypeError")
        self._report(
            gp.check._check_range(max_iterations, "max_iterations", 0.5), "ValueError"
        )
        check = gp.check._check_type(tolerance, "tolerance", [float])
        if check:
            try:
                tolerance = float(tolerance)
                self._report(
                    gp.check._conversion(tolerance, "tolerance", float), "Warning"
                )
            except Exception:
                self._report(check, "TypeError")
        self._report(
            gp.check._check_range(tolerance, "tolerance", 0.0, 1.0), "ValueError"
        )
        check = gp.check._check_type(order, "order", [int])
        if check:
            try:
                order = int(order)
                self._report(gp.check._conversion(order, "order", int), "Warning")
            except Exception:
                self._report(check, "TypeError")
        self._report(gp.check._check_value(order, "order", [1, 2]), "ValueError")
        check = gp.check._check_type(warp_0, "warp_0", [np.ndarray])
        if check:
            try:
                warp_0 = np.asarray(warp_0)
                self._report(
                    gp.check._conversion(warp_0, "warp_0", np.ndarray), "Warning"
                )
            except Exception:
                self._report(check, "TypeError")
        self._report(gp.check._check_dim(warp_0, "warp_0", 1), "ValueError")
        self._report(
            gp.check._check_axis(warp_0, "warp_0", 0, [6 * order]), "ValueError"
        )

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
        if warp_0[0] == 0 and warp_0[1] == 0:
            self._p_init = warp_0
            self._get_initial_guess()
            self._p = self._p_init
        else:
            self._p = warp_0
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
            else:
                self.solved = False
            # Pack results.
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
        except Exception:
            log.error("Subset unsolvable.")
            print(traceback.format_exc())
            self.solved = False
            self._unsolvable = True

        # Pack results.
        self.data["solved"] = self.solved
        self.data["unsolvable"] = self._unsolvable

        return self.solved

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
            self._g_img.image_gs.astype(np.float32),
            subset,
            cv2.TM_CCORR_NORMED,
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
