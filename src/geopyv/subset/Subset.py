import cv2
import os
import numpy as np
from geopyv.image import Image
from geopyv.templates import Circle, Square
from geopyv.gui import ImageSelector
from ._subset_extensions import (
    _init_reference,
    _solve_ICGN,
    _solve_FAGN,
    _solve_WFAGN,
    _f_coords,
    _Delta_f,
    _f_m,
    _grad,
    _SSSIG,
    _intensity,
    _g_coord,
    _sigma_intensity,
    _g_coords,
    _Delta_g,
    _g_m,
    _sdi,
    _hessian,
    _Delta_p_ICGN,
    _Delta_p_FAGN,
    _Delta_p_WFAGN,
    _p_new_ICGN,
    _norm,
    _ZNSSD,
    _WZNSSD,
    _D,
    _W,
    _A_s,
    _T_p,
    _dg_m_dp,
    _dW_g_dp,
    _dDelta_g_dp,
    _dg_n_dp,
    _dT_p_dp,
    _dA_s_dp,
    _grad_C_W
    )
import matplotlib.pyplot as plt

class Subset:
    """Subset class for geopyv.

    Parameters
    ----------
    coord : `numpy.ndarray` (x, y)
        Subset coordinates.
    f_img : geopyv.Image
        Reference image of geopyv.Image class, instantiated by :mod:`~image.Image`.
    g_img : geopyv.Image
        Target image of geopyv.Image class, instantiated by :mod:`~image.Image`.
    template : `geopyv.Template`
        Subset template object.


    Attributes
    ----------
    f_img : `geopyv.Image`
        Reference image of geopyv.image.Image class, instantiated by :mod:`~image.Image`.
    g_img : `geopyv.Image`
        Target image of geopyv.image.Image class, instantiated by :mod:`~image.Image`.
    template: `geopyv.Template`
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
    C_SSD : float
        Zero-normalised sum of squared differences coefficient, computed by private
        method :meth:`~_get_correlation`.
    C_CC : float
        Zero-normalised cross-correlation coefficient, computed by private method
        :meth:`~_get_correlation`.
    """

    def __init__(self, f_coord=None, f_img=None, g_img=None, template=Circle(50)):
        """Initialisation of geopyv subset object."""
        self.initialised = False
        self.f_coord = f_coord
        self.f_img = f_img
        self.g_img = g_img
        self.template = template

        # Check types.
        if type(self.f_img) != Image:
            self.f_img = self._load_f_img()
        if type(self.g_img) != Image:
            self.g_img = self._load_g_img()
        if type(self.f_coord) != np.ndarray:
            self.f_coord = np.empty(2)
            self._select_f_coord()
        elif np.shape(self.f_coord) != np.shape(np.empty(2)):
            raise TypeError("Coordinate is not an np.ndarray of length 2.")
        if type(self.template) != Circle:
            if type(template) != Square:
                raise TypeError("Template is not a type defined in geopyv.templates.")

        # Initialise subset.
        self._get_initial_guess_size()
        output = _init_reference(self.f_coord, self.template.coords, self.f_img.QCQT)
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
        self.x = self.f_coord[0]
        self.y = self.f_coord[1]

        # Check subset is entirely within the reference image.
        subset_list = np.array([
            [self.x-self.template.size, self.y-self.template.size],
            [self.x+self.template.size, self.y+self.template.size],
            [self.x-self.template.size, self.y+self.template.size],
            [self.x+self.template.size, self.y-self.template.size]
            ])
        if np.any(subset_list<0):
            raise ValueError("Subset reference partially falls outside reference image.")

    def solve(self, max_norm=1e-3, max_iterations=15, p_0=np.zeros(6), tolerance=0.75, method="ICGN"):
        """Method to solve for the subset displacements using the various methods.

        Parameters
        ----------
        max_norm : float, optional
            Exit criterion for norm of increment in warp function. Defaults to value of
            :math:`1 \cdot 10^{-5}`.
        max_iterations : int, optional
            Exit criterion for number of Gauss-Newton iterations. Defaults to value
            of 50.
        p_0 : ndarray, optional
            1D array of warp function parameters with `float` type.
        method : str
            Solution method. Options are FAGN, WFAGN and ICGN. Default is ICGN since it
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

        # Check method and length of p_0.
        if method == "ICGN" or method == "FAGN":
            if np.shape(p_0)[0] != 6 and np.shape(p_0)[0] != 12:
                raise ValueError("Invalid length of initial p_0 preconditioning vector for chosen solve method.")
        elif method == "WFAGN":
            if np.shape(p_0)[0] != 7:
                raise ValueError("Invalid length of initial p_0 preconditioning vector for chosen solve method.")

        # Store settings.
        self.method = method
        self.max_norm = max_norm
        self.max_iterations = max_iterations
        self.tolerance = tolerance

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
                results = _solve_ICGN(self.f_coord, self.f_coords, self.f, self.f_m, self.Delta_f, self.grad_f, self.f_img.QCQT, self.g_img.QCQT, self.p, self.max_norm, self.max_iterations)
            elif self.method == "FAGN" and np.mod(self.p.shape[0],2) == 0:
                results = _solve_FAGN(self.f_coord, self.f_coords, self.f, self.f_m, self.Delta_f, self.grad_f, self.f_img.QCQT, self.g_img.QCQT, self.p, self.max_norm, self.max_iterations)
            elif self.method == "WFAGN" and np.mod(self.p.shape[0],2) == 1:
                raise Exception("WFAGN method not yet fully implemented.")
                self.p[-1] = 100000
                results = _solve_WFAGN(self.f_coord, self.f_coords, self.f, self.f_m, self.Delta_f, self.grad_f, self.f_img.QCQT, self.g_img.QCQT, self.p, self.max_norm, self.max_iterations, 100)
            # Unpack results.
            self.g_coords = results[0]
            self.g = results[1]
            self.g_m = results[2][0][0]
            self.g_m = results[2][1][0]
            self.iterations = np.max(results[3][0,:]).astype(int)
            self.history = results[3][:,:self.iterations]
            self.norm = self.history[1][-1]
            self.C_CC = self.history[2][-1]
            self.C_SSD = self.history[3][-1]
            self.p = results[4]
            self.u = self.p[0][0]
            self.v = self.p[1][0]

            # Check for tolerance.
            if self.C_CC > self.tolerance:
                self.solved = True
        except:
            raise ValueError("Reporting value error...")

    def inspect(self):
        """Method to show the subset and associated quality metrics."""
        f, ax = plt.subplots(num="Subset")
        x = self.f_coord.item(0)
        y = self.f_coord.item(1)
        x_min = (np.round(x, 0) - self.template.size).astype(int)
        x_max = (np.round(x, 0) + self.template.size).astype(int)
        y_min = (np.round(y, 0) - self.template.size).astype(int)
        y_max = (np.round(y, 0) + self.template.size).astype(int)
        subset = self.f_img.image_gs.astype(np.float32)[y_min:y_max+1, x_min:x_max+1]

        # If a circular subset, mask pixels outside radius.
        if type(self.template) == Circle:
            x, y = np.meshgrid(
                np.arange(-self.template.size, self.template.size + 1, 1),
                np.arange(-self.template.size, self.template.size + 1, 1),
            )
            dist = np.sqrt(x ** 2 + y ** 2)
            mask = np.zeros(subset.shape)
            mask[dist > self.template.size] = 255
            subset = np.maximum(subset, mask)

        ax.imshow(subset, cmap="gist_gray")
        quality = r"Quality metrics: $\sigma_s$ = {:.2f}; SSSIG = {:.2E}".format(self.sigma_intensity, self.SSSIG)
        ax.text(self.template.size, 2*self.template.size + 5, quality, horizontalalignment="center")
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

    def convergence(self):
        """Method to plot the rate of convergence for the subset."""
        if hasattr(self, "history"):
            f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, num="Convergence")
            ax1.semilogy(self.history[0,:], self.history[1,:], marker="o", clip_on=False, label="Convergence")
            ax1.plot([1, self.max_iterations], [self.max_norm, self.max_norm], "--r", label="Threshold")
            ax2.plot(self.history[0,:], self.history[2,:], marker="o", clip_on=False, label="Convergence")
            ax2.plot([1, self.max_iterations], [self.tolerance, self.tolerance], "--r", label="Threshold")
            ax1.set_ylabel(r"$\Delta$ Norm (-)")
            ax1.set_ylim(self.max_norm/1000, self.max_norm*1000)
            ax1.set_yticks([self.max_norm*1000, self.max_norm*100, self.max_norm*10, self.max_norm, self.max_norm/10, self.max_norm/100, self.max_norm/1000])
            ax2.set_ylabel(r"$C_{CC}$ (-)")
            ax2.set_xlabel("Iteration number (-)")
            ax2.set_xlim(1, self.max_iterations)
            ax2.set_ylim(0.0, 1)
            ax2.set_yticks(np.linspace(0.0, 1.0, 6))
            ax2.set_xticks(np.linspace(1, self.max_iterations, self.max_iterations))
            ax1.legend(frameon=False)
            plt.tight_layout()
            plt.show()
        else:
            raise Exception("Warning: Subset not yet solved. Run the solve() method.")

    def _load_img(self, message):
        """Private method to open a file dialog and slect an image."""
        directory = os.getcwd()
        dialog = ImageSelector()
        imgpath = dialog.get_path(directory, message)
        img = Image(imgpath)
        return img

    def _load_f_img(self):
        """Private method to load the reference image."""
        print("No reference image supplied. Please select the reference image.")
        return self._load_img("Select reference image.")

    def _load_g_img(self):
        """Private method to load the target image."""
        print("No target image supplied. Please select the target image.")
        return self._load_img("Select target image.")

    def _on_select_f_coord(self, event):
        """Private method to print the selected coordinates."""
        print("Coordinate selected: {}, {}".format(self.x, self.y))

    def _on_click_f_coord(self, event):
        """Private method to store and plot the currently selected coordinate in self.f_coord."""
        if event.button==3:
            if event.xdata != None and event.ydata != None:
                if event.xdata > self.template.size and event.xdata < np.shape(self.f_img.image_gs)[0]-self.template.size:
                    if event.ydata > self.template.size and event.ydata < np.shape(self.f_img.image_gs)[1]-self.template.size:
                        self.x = np.round(event.xdata, 0)
                        self.y = np.round(event.ydata, 0)
                        self.f_coord = np.asarray([self.x, self.y])
                        ax = event.inaxes
                        f = ax.get_figure()
                        num_lines = len(ax.lines)
                        while num_lines > 0:
                            ax.lines.pop()
                            num_lines = len(ax.lines)
                        ax.plot(self.x, self.y, marker="+", color="y", zorder=10)
                        if type(self.template) == Circle:
                            theta = np.linspace(0, 2*np.pi, 150)
                            radius = self.template.size
                            x = self.x+radius*np.cos(theta)
                            y = self.y+radius*np.sin(theta)
                            ax.plot(x, y, color='y')
                        elif type(self.template) == Square:
                            x = [
                                self.x-self.template.size, 
                                self.x-self.template.size,
                                self.x+self.template.size,
                                self.x+self.template.size,
                                self.x-self.template.size,
                            ]
                            y = [
                                self.y-self.template.size, 
                                self.y+self.template.size,
                                self.y+self.template.size,
                                self.y-self.template.size,
                                self.y-self.template.size,
                            ]
                            ax.plot(x, y, color='y')
                        f.canvas.draw()
                        f.canvas.flush_events()

    def _select_f_coord(self):
        """Private method to select f_coord if not supplied by the user."""
        print("No coordinate supplied. Please select the target coordinate for the subset.")
        f, ax = plt.subplots(num="Right click on the target coordinate for the subset and close to save")
        f.canvas.mpl_connect('button_press_event', self._on_click_f_coord)
        f.canvas.mpl_connect('close_event', self._on_select_f_coord)
        ax.imshow(self.f_img.image_gs, cmap="gist_gray")
        plt.tight_layout()
        plt.show()   

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

    def pysolve(self, max_norm=1e-3, max_iterations=15, p_0=np.zeros(6), tolerance=0.75, method="ICGN"):
        # Check method and length of p_0.
        if method == "ICGN" or method == "FAGN":
            if np.shape(p_0)[0] != 6 and np.shape(p_0)[0] != 12:
                raise ValueError("Invalid length of p_0 preconditioning vector for chosen solve method.")
        elif method == "WFAGN":
            if np.shape(p_0)[0] != 7:
                raise ValueError("Invalid length of p_0 preconditioning vector for chosen solve method.")

        # Store settings.
        self.method = method
        self.max_norm = max_norm
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Compute initial guess if u and v in deformation parameter vector initialised
        # with zeros, otherwise precondition.
        if p_0[0] == 0 and p_0[1] == 0:
            self.p_init = p_0
            self._get_initial_guess()
            self.p = self.p_init
        else:
            self.p = p_0
    
        self.iterations = 0
        self.norm = 1
        self.size = len(self.f_coords)**0.5
    
        try:
            if method == "ICGN":
                # Compute reference quantities.
                self._get_sdi_f()
                self._get_hessian()
                # Iterate to solution.
                while self.iterations<self.max_iterations and self.norm>self.max_norm:
                    self._get_g_coord()
                    self._get_g_coords()
                    self._get_intensity_g()
                    self._get_g_m()
                    self._get_Delta_g()
                    self._get_Delta_p_ICGN()
                    self._get_p_new_ICGN()
                    self._get_norm()
                    self._get_ZNSSD()
                    self.C_CC = 1 - (self.C_SSD/2)
                    self.iterations += 1
                    self.p = self.p_new
                    self.u = self.p[0]
                    self.v = self.p[1]
                    if hasattr(self, "history"):
                        self.history = np.hstack((self.history, np.array([[self.iterations], [self.norm], [self.C_CC]], ndmin=2)))
                    else:
                        self.history = np.array([[self.iterations], [self.norm], [self.C_CC]], ndmin=2)
            elif method == "FAGN":
                # Iterate to solution.
                while self.iterations<self.max_iterations and self.norm>self.max_norm:
                    self._get_g_coord()
                    self._get_g_coords()
                    self._get_intensity_g()
                    self._get_g_m()
                    self._get_Delta_g()
                    self._get_grad_g()
                    self._get_sdi_g()
                    self._get_hessian()
                    self._get_Delta_p_FAGN()
                    self.p_new = self.p + self.Delta_p
                    self._get_norm()
                    self._get_ZNSSD()
                    self.C_CC = 1 - (self.C_SSD/2)
                    self.iterations += 1
                    self.p = self.p_new
                    self.u = self.p[0]
                    self.v = self.p[1]
                    if hasattr(self, "history"):
                        self.history = np.hstack((self.history, np.array([[self.iterations], [self.norm], [self.C_CC]], ndmin=2)))
                    else:
                        self.history = np.array([[self.iterations], [self.norm], [self.C_CC]], ndmin=2)
            elif method == "WFAGN":
                raise Exception("WFAGN method not yet fully implemented.")
                # Compute reference quantities.

                self._get_D_f()
                self.D_0_min = 10
                # Iterate to solution.
                while self.iterations<self.max_iterations and self.norm>self.max_norm:
                    self._get_W_f()
                    self._get_A_s()
                    self._get_g_coord()
                    self._get_g_coords()
                    self._get_intensity_g()
                    self._get_g_m()
                    self._get_Delta_g()
                    self._get_grad_g()
                    self._get_D_g()
                    self._get_W_g()
                    self._get_sdi_g()
                    self._get_T_p()
                    self._get_dg_m_dp()
                    self._get_dW_g_dp()
                    self._get_dDelta_g_dp()
                    self._get_dg_n_dp()
                    self._get_dT_p_dp()
                    self._get_hessian_dT_p_dp()
                    self.hessian *= 2/self.A_s
                    self._get_dA_s_dp()
                    self._get_grad_C_W()
                    self._get_Delta_p_WFAGN()
                    self.p_new = self.p - self.Delta_p
                    self._get_norm()
                    self._get_WZNSSD()
                    self.C_CC = 1 - (self.C_SSD/2)
                    self.iterations += 1
                    self.p = self.p_new
                    self.u = self.p[0]
                    self.v = self.p[1]
                    
                    if self.p[-1] < self.D_0_min:
                        self.p[-1] = self.D_0_min
                    elif self.p[-1] > p_0[-1]:
                        self.p[-1] = p_0[-1]

                    if hasattr(self, "history"):
                        self.history = np.hstack((self.history, np.array([[self.iterations], [self.norm], [self.C_CC]], ndmin=2)))
                    else:
                        self.history = np.array([[self.iterations], [self.norm], [self.C_CC]], ndmin=2)
        
            # Check for tolerance.
            if self.C_CC > self.tolerance:
                self.solved = True
        except:
            self.unsolvable = True

    # Aliases to C++ methods.
    def _get_f_coords(self):
        self.f_coords = _f_coords(self.coord, self.template_coords)
    def _get_Delta_f(self):
        self.Delta_f = _Delta_f(self.f, self.f_m)
    def _get_f_m(self):
        self.f_m = _f_m(self.f)
    def _get_grad_f(self):
        self.grad_f = _grad(self.f_coords, self.f_img.QCQT)
    def _get_grad_g(self):
        self.grad_g = _grad(self.g_coords, self.g_img.QCQT)
    def _get_SSSIG(self):
        self.SSSIG = _SSSIG(self.f_coords, self.grad_f)
    def _get_intensity_f(self):
        self.f = _intensity(self.f_coords, self.f_img.QCQT)
    def _get_intensity_g(self):
        self.g = _intensity(self.g_coords, self.g_img.QCQT)
    def _get_g_coord(self):
        self.g_coord = _g_coord(self.f_coord, self.p)
    def _get_sigma_intensity(self):
        self.sigma = _sigma_intensity(self.f, self.f_m)
    def _get_g_coords(self):
        self.g_coords = _g_coords(self.f_coord, self.p, self.f_coords)
    def _get_Delta_g(self):
        self.Delta_g = _Delta_g(self.g, self.g_m)
    def _get_g_m(self):
        self.g_m = _g_m(self.g)
    def _get_sdi_f(self):
        self.sdi = _sdi(self.f_coord, self.f_coords, self.grad_f, self.p)
    def _get_sdi_g(self):
        self.sdi = _sdi(self.g_coord, self.g_coords, self.grad_g, self.p)
    def _get_hessian(self):
        self.hessian = _hessian(self.sdi)
    def _get_hessian_dT_p_dp(self):
        self.hessian = _hessian(self.dT_p_dp)
    def _get_Delta_p_ICGN(self):
        self.Delta_p = _Delta_p_ICGN(self.hessian, self.f, self.g, self.f_m, self.g_m, self.Delta_f, self.Delta_g, self.sdi)
    def _get_Delta_p_FAGN(self):
        self.Delta_p = _Delta_p_FAGN(self.hessian, self.f, self.g, self.f_m, self.g_m, self.Delta_f, self.Delta_g, self.sdi)
    def _get_Delta_p_WFAGN(self):
        self.Delta_p = _Delta_p_WFAGN(self.hessian, self.grad_C_W)
    def _get_p_new_ICGN(self):
        self.p_new = _p_new_ICGN(self.p, self.Delta_p)
    def _get_norm(self):
        self.norm = _norm(self.Delta_p, self.size)
    def _get_ZNSSD(self):
        self.C_SSD = _ZNSSD(self.f, self.g, self.f_m, self.g_m, self.Delta_f, self.Delta_g)
    def _get_WZNSSD(self):
        self.C_SSD = _WZNSSD(self.f, self.g, self.f_m, self.g_m, self.Delta_f, self.Delta_g, self.W_f, self.W_g, self.A_s)
    def _get_D_f(self):
        self.D_f = _D(self.f_coord, self.f_coords)
    def _get_D_g(self):
        self.D_g = _D(self.g_coord, self.g_coords)
    def _get_W_f(self):
        self.W_f = _W(self.D_f, self.p[-1])
    def _get_W_g(self):
        self.W_g = _W(self.D_g, self.p[-1])
    def _get_A_s(self):
        self.A_s = _A_s(self.W_f)
    def _get_T_p(self):
        self.T_p = _T_p(self.f, self.g, self.f_m, self.g_m, self.Delta_f, self.Delta_g, self.W_f, self.W_g)
    def _get_dg_m_dp(self):
        self.dg_m_dp = _dg_m_dp(self.sdi)
    def _get_dW_g_dp(self):
        self.dW_g_dp = _dW_g_dp(self.f_coord, self.f_coords, self.W_g, self.p)
    def _get_dDelta_g_dp(self):
        self.dDelta_g_dp = _dDelta_g_dp(self.g, self.g_m, self.Delta_g, self.sdi, self.dg_m_dp)
    def _get_dg_n_dp(self):
        self.dg_n_dp = _dg_n_dp(self.g, self.g_m, self.Delta_g, self.sdi, self.dg_m_dp, self.dDelta_g_dp)
    def _get_dT_p_dp(self):
        self.dT_p_dp = _dT_p_dp(self.f, self.g, self.f_m, self.g_m, self.Delta_f, self.Delta_g, self.p, self.W_f, self.W_g, self.D_f, self.D_g, self.dg_n_dp, self.dW_g_dp)
    def _get_dA_s_dp(self):
        self.dA_s_dp = _dA_s_dp(self.W_f, self.D_f, self.p)
    def _get_grad_C_W(self):
        self.grad_C_W = _grad_C_W(self.f, self.g, self.f_m, self.g_m, self.Delta_f, self.Delta_g, self.p, self.W_f, self.W_g, self.D_f, self.D_g, self.T_p, self.dT_p_dp, self.dA_s_dp, self.A_s)

    def results(self):
        """Method to print the results"""
        if self.solved: 
            print("Initial horizontal coordinate: {x_i} (px); Initial vertical coordinate: {y_i} (px)".format(x_i=self.x, y_i=self.y))
            print("Horizontal displacement: {u} (px); Vertical displacement: {v} (px)".format(u=self.u, v=self.v))
            print("Correlation coefficient: CC = {C_CC} (-), SSD = {C_SSD} (-)".format(C_CC=self.C_CC, C_SSD=self.C_SSD))
            print("Final horizontal coordinate: {x_f} (px); Final vertical coordinate: {y_f} (px)".format(x_f=self.x+self.u, y_f=self.y+self.v))
        else:
            print("Subset not yet solved. Use command `subset.solve()`.")
    
    