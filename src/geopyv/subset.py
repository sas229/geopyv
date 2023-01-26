import logging 
import cv2
import os
import numpy as np
import geopyv as gp

log = logging.getLogger(__name__)

class SubsetBase:
    """Subset base class to be used as a mixin."""
    def inspect(self, show=True, block=True, save=None):
        """Method to show the subset and associated quality metrics."""
        gp.plots.inspect_subset(self.data, mask=None, show=show, block=block, save=save)

    def convergence(self, show=True, block=True, save=None):
        """Method to plot the rate of convergence for the subset."""
        if "results" in self.data:
            gp.plots.convergence_subset(self.data, show=show, block=block, save=save)
        else:
            raise Exception("Warning: Subset not yet solved. Run the solve() method.")

class Subset(SubsetBase):
    """Subset class for geopyv.

    Parameters
    ----------
    coord : `numpy.ndarray` (x, y)
        Subset coordinates.
    f_img : geopyv.image.Image
        Reference image of geopyv.image.Image class, instantiated by :mod:`~image.Image`.
    g_img : geopyv.image.Image
        Target image of geopyv.imageImage class, instantiated by :mod:`~image.Image`.
    template : `geopyv.templates.Template`
        Subset template object.

    Attributes
    ----------
    f_img : `geopyv.image.Image`
        Reference image of geopyv.image.Image class, instantiated by :mod:`~image.Image`.
    g_img : `geopyv.image.Image`
        Target image of geopyv.image.Image class, instantiated by :mod:`~image.Image`.
    template: `geopyv.templates.Template`
        Subset template object.
    method : `str`
        Solver type. Options are 'ICGN' and 'FAGN'.
    init_guess_size : int
        Size of subset used to define the initial guess, approximated by private method
        :meth:`~_get_initial_guess_size`.
    f_coord : `numpy.ndarray` (x, y)
        1D array of the coordinates of the subset in reference image of type `float`.
    f_coords : `numpy.ndarray` (Nx, 2)
        2D array of subset coordinates in reference image of type `float`.
    grad_f : `numpy.ndarray` (Nx, 2)
        Gradients of reference image `f`.
    SSSIG : float
        Sum of the square of the reference subset intensity gradients.
    sigma_intensity : float
        Standard deviaition of the reference subset intensities.
    p_0 : `numpy.ndarray` (Nx, 1)
        1D array of initial warp function parameters of type `float`, used to precondition
        class method :meth:`~solve`.
    p : `numpy.ndarray` (Nx, 1)
        1D array of warp function parameters of type `float`, output by class
        method :meth:`~solve`.
    norm : float
        Custom norm of the increment in the warp function parameters after
        Gao et al. (2015), computed by private method :meth:`~_get_norm`.
    C_ZNSSD : float
        Zero-normalised sum of squared differences coefficient, computed by private
        method :meth:`~_get_correlation`.
    C_ZNCC : float
        Zero-normalised cross-correlation coefficient, computed by private method
        :meth:`~_get_correlation`.
    x : float
        Initial horizontal coordinate.
    y : float 
        Initial vertical coordinate.
    u : float
        Horizontal displacement.
    v : float 
        Vertical displacement.
    x_f : float
        Final horizontal coordinate.self.initialised
    y_f : float
        Final vertical coordinate.
    settings: dict
        Dictionary of settings.
    quality: dict 
        Dictionary of image quality measures.
    results: dict
        Dictionary of results.
    """

    def __init__(self, *, f_coord=None, f_img=None, g_img=None, template=None):
        """Initialisation of geopyv subset object."""
        self.initialised = False
        self.f_coord = f_coord
        self.f_img = f_img
        self.g_img = g_img
        self.template = template

        # Check types.
        if type(self.f_img) != gp.image.Image:
            self.f_img = self._load_f_img()
        if type(self.g_img) != gp.image.Image:
            self.g_img = self._load_g_img()
        if type(self.f_coord) != np.ndarray:
            self.f_coord = np.empty(2)
            coordinate = gp.gui.CoordinateSelector()
            self.f_coord = coordinate.select(self.f_img, self.template)
        elif np.shape(self.f_coord) != np.shape(np.empty(2)):
            raise TypeError("Coordinate is not an np.ndarray of length 2.")
        if self.template == None:
            self.template = gp.templates.Circle(50)
        elif type(self.template) != gp.templates.Circle and type(self.template) != gp.templates.Square:
            raise TypeError("Template is not a type defined in geopyv.templates.")

        # Check subset is entirely within the reference image.
        self.x = self.f_coord[0]
        self.y = self.f_coord[1]
        subset_list = np.array([
            [self.x-self.template.size, self.y-self.template.size],
            [self.x+self.template.size, self.y+self.template.size],
            [self.x-self.template.size, self.y+self.template.size],
            [self.x+self.template.size, self.y-self.template.size]
            ])
        if np.any(subset_list<0):
            raise ValueError("Subset reference partially falls outside reference image.")

        # Initialise subset.
        self._get_initial_guess_size()
        output = gp._subset_extensions._init_reference(self.f_coord, self.template.coords, self.f_img.QCQT)
        self.f_coords = output[0]
        self.f = output[1]
        self.f_m = output[2][0][0]
        self.Delta_f = output[2][1][0]
        self.grad_f = output[3]
        self.SSSIG = output[4][0][0]
        self.sigma_intensity = output[4][1][0]
        self.solved = False
        self.unsolvable = False
        self.initialised == True
        self.quality = {
            "SSSIG": self.SSSIG,
            "sigma_intensity": self.sigma_intensity,
        }

        # Data.
        self.data = {
            "type": "Subset",
            "solved": self.solved,
            "unsolvable": self.unsolvable,
            "images": {
                "f_img": self.f_img.filepath,
                "g_img": self.g_img.filepath,
            },
            "position": {
                "x": self.x,
                "y": self.y,
            },
            "quality": self.quality,
            "template": {
                "shape": self.template.shape,
                "dimension": self.template.dimension,
                "size": self.template.size,
                "n_px": self.template.n_px,
            }
        }

    def solve(self, *, max_norm=1e-3, max_iterations=15, order=1, p_0=None, tolerance=0.7, method="ICGN"):
        """Method to solve for the subset displacements using the various methods.

        Parameters
        ----------
        max_norm : float, optional
            Exit criterion for norm of increment in warp function. Defaults to value of
            :math:`1 \cdot 10^{-5}`.
        max_iterations : int, optional
            Exit criterion for number of Gauss-Newton iterations. Defaults to value
            of 50.
        order : int
            Warp function order. Options are 1 and 2.
        p_0 : ndarray, optional
            1D array of warp function parameters with `float` type.
        method : str
            Solution method. Options are FAGN and ICGN. Default is ICGN since it
            is faster.


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
        """

        # Check other control parameters.
        if max_norm < 1e-10:
            raise ValueError("Maximum norm value too small. Suggested default is 1e-3.")
        elif type(max_iterations) != int:
            raise TypeError("Maximum number of iterations of invalid type. Must be positive integer.")
        elif max_iterations <= 0:
            raise ValueError("Invalid maximum number of iterations specified. Must be positive integer.") 
        elif type(tolerance) != float:
            raise TypeError("Tolerance of invalid type. Must be float greater than zero and less than one.")
        elif tolerance < 0:
            raise ValueError("Tolerance must be greater than or equal to zero. Suggested default is 0.75.")
        elif tolerance > 1:
            raise ValueError("Tolerance must be less than one. Suggested default is 0.75.")

        # Check warp function order and type.
        if type(p_0) == type(None):
            if order == 1:
                p_0 = np.zeros(6)
            elif order == 2:
                p_0 = np.zeros(12)
            else:
                raise ValueError("Invalid warp function order.")
        else:
            if type(p_0) != np.ndarray:
                raise TypeError("Warp function of incorrect type.")

        # Store settings.
        self.method = method
        self.order = order
        self.max_norm = max_norm
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.settings = {
            "method": self.method,
            "order": self.order,
            "max_norm": self.max_norm,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
        }
        self.data.update({"settings": self.settings})

        # Compute initial guess if u and v in deformation parameter vector initialised
        # with zeros, otherwise precondition.
        if p_0[0] == 0 and p_0[1] == 0:
            self.p_init = p_0
            self._get_initial_guess()
            self.p = self.p_init
        else:
            self.p = p_0

        # Call appropriate iterative solver:
        try:
            if self.method == "ICGN" and np.mod(self.p.shape[0],2) == 0:
                results = gp._subset_extensions._solve_ICGN(self.f_coord, self.f_coords, self.f, self.f_m, self.Delta_f, self.grad_f, self.f_img.QCQT, self.g_img.QCQT, self.p, self.max_norm, self.max_iterations)
            elif self.method == "FAGN" and np.mod(self.p.shape[0],2) == 0:
                results = gp._subset_extensions._solve_FAGN(self.f_coord, self.f_coords, self.f, self.f_m, self.Delta_f, self.grad_f, self.f_img.QCQT, self.g_img.QCQT, self.p, self.max_norm, self.max_iterations)
            # Unpack results.
            self.g_coords = results[0]
            self.g = results[1]
            self.g_m = results[2][0][0]
            self.g_m = results[2][1][0]
            self.iterations = np.max(results[3][0,:]).astype(int)
            self.history = results[3][:,:self.iterations]
            self.norm = self.history[1][-1]
            self.C_ZNCC = self.history[2][-1]
            self.C_ZNSSD = self.history[3][-1]
            self.p = results[4]
            self.u = self.p[0][0]
            self.v = self.p[1][0]
            self.x_f = self.x+self.u
            self.y_f = self.y+self.v

            # Check for tolerance.
            if self.C_ZNCC > self.tolerance:
                self.solved = True
                log.debug("Subset solved.")
                log.debug("Initial horizontal coordinate: {x_i} (px); Initial vertical coordinate: {y_i} (px)".format(x_i=self.x, y_i=self.y))
                log.debug("Horizontal displacement: {u} (px); Vertical displacement: {v} (px)".format(u=self.u, v=self.v))
                log.debug("Correlation coefficient: CC = {C_ZNCC} (-), SSD = {C_ZNSSD} (-)".format(C_ZNCC=self.C_ZNCC, C_ZNSSD=self.C_ZNSSD))
                log.debug("Final horizontal coordinate: {x_f} (px); Final vertical coordinate: {y_f} (px)".format(x_f=self.x_f, y_f=self.y_f))
        
            # Pack results.
            self.data["solved"] = self.solved
            self.data["unsolvable"] = self.unsolvable
            self.results = {
                "u": self.u,
                "v": self.v,
                "p": self.p,
                "history": self.history,
                "iterations": self.iterations,
                "norm": self.norm,
                "C_ZNCC": self.C_ZNCC,
                "C_ZNSSD": self.C_ZNSSD,
            }
            self.data.update({"results": self.results})
            
            # Return solved boolean.
            return self.solved

        except:
            raise RuntimeError("Runtime error in solve method.")

    def _load_img(self, message):
        """Private method to open a file dialog and select an image."""
        directory = os.getcwd()
        print(gp.gui.ImageSelector)
        dialog = gp.gui.ImageSelector()
        imgpath = dialog.get_path(directory, message)
        img = gp.image.Image(imgpath)
        return img

    def _load_f_img(self):
        """Private method to load the reference image."""
        log.warn("No reference image supplied. Please select the reference image.")
        return self._load_img("Select reference image.")

    def _load_g_img(self):
        """Private method to load the target image."""
        log.warn("No target image supplied. Please select the target image.")
        return self._load_img("Select target image.")

    def _get_initial_guess_size(self):
        """Private method to estimate the size of square subset to use in the
        initial guess."""
        self.initial_guess_size = np.round(np.sqrt(self.template.n_px), 1)

    def _get_initial_guess(self):
        """Private method to compute an initial guess of the subset displacement using
        OpenCV function :py:meth:`cv2.matchTemplate` and the Normalised
        Cross-Correlation (NCC) criteria."""
        # Extract square subset for initial guess.
        x = self.f_coord.item(0)
        y = self.f_coord.item(1)
        x_min = (np.round(x, 0) - self.initial_guess_size / 2).astype(int)
        x_max = (np.round(x, 0) + self.initial_guess_size / 2).astype(int)
        y_min = (np.round(y, 0) - self.initial_guess_size / 2).astype(int)
        y_max = (np.round(y, 0) + self.initial_guess_size / 2).astype(int)
        subset = self.f_img.image_gs.astype(np.float32)[y_min:y_max, x_min:x_max]

        # Apply template matching technique.
        res = cv2.matchTemplate(
            self.g_img.image_gs.astype(np.float32), subset, cv2.TM_CCORR_NORMED
        )
        max_loc = cv2.minMaxLoc(res)[3]

        # Create initialised warp vector with affine displacements preconditioned.
        self.p_init[0] = (max_loc[0] + self.initial_guess_size / 2) - x
        self.p_init[1] = (max_loc[1] + self.initial_guess_size / 2) - y
  
class SubsetResults(SubsetBase):
    """SubsetResults class for geopyv.

    Parameters
    ----------
    data : dict
        geopyv data dict from Subset object.

    Attributes
    ----------
    data : dict
        geopyv data dict from Subset object.
    """

    def __init__(self, data):
        """Initialisation of geopyv SubsetResults class."""
        self.data = data