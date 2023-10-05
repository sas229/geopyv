"""

Particle module for geopyv.

"""
import logging
import numpy as np
import geopyv as gp
import scipy as sp
from geopyv.object import Object
import re
from alive_progress import alive_bar
import math
import geomat
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


class ParticleBase(Object):
    """
    Particle base class to be used as a mixin. Contains plot functionality.
    """

    def __init__(self):
        super().__init__(object_type="Particle")
        """

        Particle base class initialiser.

        """

    def trace(
        self,
        quantity="warps",
        component=0,
        imshow=True,
        colorbar=True,
        ticks=None,
        alpha=0.75,
        axis=True,
        xlim=None,
        ylim=None,
        show=True,
        block=True,
        save=None,
    ):
        """
        Method to plot an incremental quantity along the particle position path.

        Parameters
        ----------
        quantity : str, optional
            Specifier for which metric to plot along the particle path.
        component : int, optional
            Specifier for which component of the metric to plot along the particle path.
        imshow : bool, optional
            Control whether the reference image is plotted.
            Defaults to True.
        colorbar : bool, optional
            Control whether the colour bar is plotted.
            Defaults to True.
        ticks : list, optional
            Overwrite default colourbar ticks.
            Defaults to None.
        alpha : float, optional
            Control contour opacity. Must be between 0.0-1.0.
            Defaults to 0.75.
        axis : bool, optional
            Control whether the axes are plotted.
            Defaults to True.
        xlim : array-like, optional
            Set the plot x-limits (lower_limit,upper_limit).
            Defaults to None.
        ylim : array-like, optional
            Set the plot y-limits (lower_limit,upper_limit).
            Defaults to None.
        show : bool, optional
            Control whether the plot is displayed.
            Defaults to True.
        block : bool, optional
            Control whether the plot blocks execution until closed.
            Defaults to False.
        save : str, optional
            Name to use to save plot. Uses default extension of `.png`.
        """

        # Check if solved.
        if self.data["solved"] is not True:
            log.error(
                "Particle not yet solved therefore no convergence data to plot. "
                "First, run :meth:`~geopyv.particle.Particle.solve()` to solve."
            )
            raise ValueError(
                "Particle not yet solved therefore no convergence data to plot. "
                "First, run :meth:`~geopyv.particle.Particle.solve()` to solve."
            )

        # Check input.
        self._report(
            gp.check._check_type(quantity, "quantity", [str, type(None)]), "TypeError"
        )
        if quantity:
            self._report(
                gp.check._check_value(
                    quantity,
                    "quantity",
                    [
                        "coordinates",
                        "warps",
                        "volumes",
                        "stresses",
                    ],
                ),
                "ValueError",
            )
        self._report(gp.check._check_type(component, "component", [int]), "TypeError")
        self._report(
            gp.check._check_index(
                component, "component", 1, self.data["results"][quantity]
            ),
            "IndexError",
        )
        self._report(gp.check._check_type(imshow, "imshow", [bool]), "TypeError")
        self._report(gp.check._check_type(colorbar, "colorbar", [bool]), "TypeError")
        types = [tuple, list, np.ndarray, type(None)]
        self._report(gp.check._check_type(ticks, "ticks", types), "TypeError")
        check = gp.check._check_type(alpha, "alpha", [float])
        if check:
            try:
                alpha = float(alpha)
                self._report(gp.check._conversion(alpha, "alpha", float), "Warning")
            except Exception:
                self._report(check, "TypeError")
        self._report(gp.check._check_range(alpha, "alpha", 0.0, 1.0), "ValueError")
        self._report(gp.check._check_type(axis, "axis", [bool]), "TypeError")
        types = [tuple, list, np.ndarray, type(None)]
        self._report(gp.check._check_type(xlim, "xlim", types), "TypeError")
        if xlim is not None:
            self._report(gp.check._check_dim(xlim, "xlim", 1), "ValueError")
            self._report(gp.check._check_axis(xlim, "xlim", 0, [2]), "ValueError")
        self._report(gp.check._check_type(ylim, "ylim", types), "TypeError")
        if ylim is not None:
            self._report(gp.check._check_dim(ylim, "ylim", 1), "ValueError")
            self._report(gp.check._check_axis(ylim, "ylim", 0, [2]), "ValueError")
        self._report(gp.check._check_type(show, "show", [bool]), "TypeError")
        self._report(gp.check._check_type(block, "block", [bool]), "TypeError")
        self._report(gp.check._check_type(save, "save", [str, type(None)]), "TypeError")

        fig, ax = gp.plots.trace_particle(
            data=self.data,
            quantity=quantity,
            component=component,
            obj_type="Particle",
            imshow=imshow,
            colorbar=True,
            ticks=ticks,
            alpha=alpha,
            axis=axis,
            xlim=xlim,
            ylim=ylim,
            show=show,
            block=block,
            save=save,
        )
        return fig, ax

    def history(
        self,
        quantity="warps",
        components=None,
        xlim=None,
        ylim=None,
        show=True,
        block=True,
        save=None,
    ):
        """
        Method to plot particle time history.

        Parameters
        ----------
        quantity : str, optional
            Specifier for which metric to plot along the particle path.
        component : int, optional
            Specifier for which components of the metric to plot.
        xlim : array-like, optional
            Set the plot x-limits (lower_limit,upper_limit).
            Defaults to None.
        ylim : array-like, optional
            Set the plot y-limits (lower_limit,upper_limit).
            Defaults to None.
        show : bool, optional
            Control whether the plot is displayed.
            Defaults to True.
        block : bool, optional
            Control whether the plot blocks execution until closed.
            Defaults to False.
        save : str, optional
            Name to use to save plot. Uses default extension of `.png`.
        """

        # Check if solved.
        if self.data["solved"] is not True:
            log.error(
                "Particle not yet solved therefore no convergence data to plot. "
                "First, run :meth:`~geopyv.particle.Particle.solve()` to solve."
            )
            raise ValueError(
                "Particle not yet solved therefore no convergence data to plot. "
                "First, run :meth:`~geopyv.particle.Particle.solve()` to solve."
            )

        # Check input.
        self._report(
            gp.check._check_type(quantity, "quantity", [str, type(None)]), "TypeError"
        )
        if quantity:
            self._report(
                gp.check._check_value(
                    quantity,
                    "quantity",
                    [
                        "coordinates",
                        "warps",
                        "volumes",
                        "stresses",
                        "works",
                    ],
                ),
                "ValueError",
            )
        self._report(
            gp.check._check_type(components, "components", [list, type(None)]),
            "TypeError",
        )
        if components:
            for component in components:
                self._report(
                    gp.check._check_index(
                        component, "component", 1, self.data["results"][quantity]
                    ),
                    "IndexError",
                )
        types = [tuple, list, np.ndarray, type(None)]
        self._report(gp.check._check_type(xlim, "xlim", types), "TypeError")
        if xlim is not None:
            self._report(gp.check._check_dim(xlim, "xlim", 1), "ValueError")
            self._report(gp.check._check_axis(xlim, "xlim", 0, [2]), "ValueError")
        self._report(gp.check._check_type(ylim, "ylim", types), "TypeError")
        if ylim is not None:
            self._report(gp.check._check_dim(ylim, "ylim", 1), "ValueError")
            self._report(gp.check._check_axis(ylim, "ylim", 0, [2]), "ValueError")
        self._report(gp.check._check_type(show, "show", [bool]), "TypeError")
        self._report(gp.check._check_type(block, "block", [bool]), "TypeError")
        self._report(gp.check._check_type(save, "save", [str, type(None)]), "TypeError")

        fig, ax = gp.plots.history_particle(
            data=self.data,
            quantity=quantity,
            components=components,
            xlim=xlim,
            ylim=ylim,
            show=show,
            block=block,
            save=save,
        )
        return fig, ax

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


class Particle(ParticleBase):
    """Particle class for geopyv.

    Private Attributes
    ------------------

    """

    def __init__(
        self,
        *,
        series=None,
        coordinate_0=np.zeros(2),
        warp_0=np.zeros(12),
        stress_0=np.zeros(6),
        volume_0=1.0,
        track=True,
    ):
        """Initialisation of geopyv particle object.

        Parameters
        ----------
        meshes : `numpy.ndarray` of geopyv.mesh.Mesh objects
            Sequence for the particle object to track.
        coordinate_0 : numpy.ndarray (2)
            Initial particle coordinate (x,y)
        p_init : `numpy.ndarray` (12), optional
            Initial warp vector.
        vol : float
            Volume represented by the particle.
        track : bool
            Boolean for Lagrangian (False) or Eulerian (True) specification.
            Defaults to False.
        """

        # Set initialised boolean.
        self._initialised = False

        # Check inputs.
        self._report(
            gp.check._check_type(
                series,
                "series",
                [
                    gp.sequence.Sequence,
                    gp.sequence.SequenceResults,
                    gp.mesh.Mesh,
                    gp.mesh.MeshResults,
                ],
            ),
            "TypeError",
        )

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
        self._report(gp.check._check_axis(warp_0, "warp_0", 0, [6, 12]), "ValueError")
        check = gp.check._check_type(
            volume_0, "volume_0", [float, np.floating, np.float64, np.float32]
        )
        if check:
            try:
                volume_0 = float(volume_0)
                self._report(
                    gp.check._conversion(volume_0, "volume_0", float), "Warning"
                )
            except Exception:
                self._report(check, "TypeError")
        self._report(gp.check._check_range(volume_0, "volume_0", 0.0), "ValueError")
        self._report(gp.check._check_type(track, "track", [bool]), "TypeError")

        if series.data["type"] == "Sequence":
            self._series_type = "Sequence"
            if "file_settings" in series.data:
                if series.data["file_settings"]["save_by_reference"]:
                    mesh_no = np.shape(series.data["meshes"])[0]
                    self._series = np.empty(mesh_no, dtype=object)
                    with alive_bar(
                        mesh_no, dual_line=True, bar="blocks", title="Loading meshes..."
                    ) as bar:
                        for i in range(mesh_no):
                            self._series[i] = gp.io.load(
                                filename=series.data["file_settings"]["mesh_dir"]
                                + series.data["meshes"][i],
                                verbose=False,
                            ).data
                            bar()
                else:
                    self._series = series.data["meshes"]
            else:
                self._series = series.data["meshes"]
        else:
            self._series_type = "Mesh"
            self._series = np.asarray([series.data])
        self._mesh_order = self._series[0]["mesh_order"]
        self._track = track
        if self._report(
            gp.check._check_type(coordinate_0, "coordinate_0", [np.ndarray]), "Warning"
        ):
            selector = gp.gui.selectors.coordinate.CoordinateSelector()
            f_img = gp.image.Image(filepath=self._series[0]["images"]["f_img"])
            coordinate_0 = selector.select(f_img, gp.templates.Circle(20))
        elif self._report(
            gp.check._check_dim(coordinate_0, "coordinate_0", 1), "Warning"
        ):
            selector = gp.gui.selectors.coordinate.CoordinateSelector()
            f_img = gp.image.Image(filepath=self._series[0]["images"]["f_img"])
            coordinate_0 = selector.select(f_img, gp.templates.Circle(20))
        elif self._report(
            gp.check._check_axis(coordinate_0, "coordinate_0", 0, [2]), "Warning"
        ):
            selector = gp.gui.selectors.coordinate.CoordinateSelector()
            f_img = gp.image.Image(filepath=self._series[0]["images"]["f_img"])
            coordinate_0 = selector.select(f_img, gp.templates.Circle(20))

        if self._series[0]["mask"][int(coordinate_0[1]), int(coordinate_0[0])] == 0:
            log.warning(
                "`coordinate_0` keyword argument value out of boundary.\n"
                "Select `coordinate_0`..."
            )
            selector = gp.gui.selectors.coordinate.CoordinateSelector()
            f_img = gp.image.Image(filepath=self._series[0]["images"]["f_img"])
            coordinate_0 = selector.select(f_img, gp.templates.Circle(20))

        self._coordinates = np.zeros((len(self._series) + 1, 2))
        self._warps = np.zeros((len(self._series) + 1, 6 * self._mesh_order))
        self._volumes = np.zeros(len(self._series) + 1)
        self._stresses = np.zeros((len(self._series) + 1, 6))
        self._works = np.zeros(len(self._series) + 1)
        self._strains = np.zeros((len(self._series) + 1, 6))

        self._coordinates[0] = coordinate_0
        self._warps[0] = warp_0[: np.shape(self._warps)[1]]
        self._volumes[0] = volume_0
        self._stresses[0] = stress_0

        self._reference_index = 0
        self.solved = False
        self._calibrated = series.data["calibrated"]

        self._initialised = True
        self.data = {
            "type": "Particle",
            "solved": self.solved,
            "series_type": self._series_type,
            "track": self._track,
            "coordinate_0": self._coordinates[0],
            "warp_0": self._warps[0],
            "volume_0": self._volumes[0],
            "image_0": self._series[0]["images"]["f_img"],
        }

    def solve(
        self, *, model=None, state=None, parameters=None
    ):  # model, state, parameters):
        """

        Method to solve for the particle.

        """
        self.solved += self._strain_path()
        if model:
            self._ps = np.zeros((len(self._series) + 1))
            self._qs = np.zeros((len(self._series) + 1))
            self._states = np.zeros((len(self._series) + 1, len(state)))
            self.solved += self._stress_path(model, state, parameters)
            self._results = {
                "coordinates": self._coordinates,
                "warps": self._warps,
                "volumes": self._volumes,
                "stresses": self._stresses,
                "mean_effective_stresses": self._ps,
                "deviatoric_stresses": self._qs,
                "states": self._states,
                "works": self._works,
            }
        else:
            self._results = {
                "coordinates": self._coordinates,
                "warps": self._warps,
                "volumes": self._volumes,
            }
        self.solved = bool(self.solved)
        self.data.update(
            {
                "parameters": parameters,
                "state": state,
                "results": self._results,
                "reference_update_register": self._reference_update_register,
                "solved": self.solved,
            }
        )

        return self.solved

    def _element_locator(self, m):
        """

        Private method to identify the current element index
        for the particle.

        Parameters
        ----------
        m : int
            Series index.

        """
        elements = self._series[m]["elements"]
        nodes = self._series[m]["nodes"]
        centroids = self._series[m]["centroids"]
        flag = False
        diff = centroids - self._coordinates[self._reference_index]
        dist = np.einsum("ij,ij->i", diff, diff)
        dist_sorted = np.argpartition(dist, min(10, math.ceil(0.05 * len(dist))))
        for index in dist_sorted:
            hull = sp.spatial.Delaunay(nodes[elements[index]][:3])
            if hull.find_simplex(self._coordinates[self._reference_index]) >= 0:
                flag = True
                break
        if flag is False:
            log.error(
                "Particle {particle} is outside of boundary {boundary}.\n"
                " Check for convex boundary update.".format(
                    particle=self._coordinates[self._reference_index],
                    boundary=self._series[m]["nodes"][self._series[m]["boundary"]],
                )
            )
            fig, ax = plt.subplots()
            ax.plot(
                self._series[m]["nodes"][self._series[m]["boundary"]][
                    [0, 1, 2, 3, 0], 0
                ],
                self._series[m]["nodes"][self._series[m]["boundary"]][
                    [0, 1, 2, 3, 0], 1
                ],
            )

            ax.plot(
                self._series[m]["nodes"][self._series[m]["exclusions"][0]][:, 0],
                self._series[m]["nodes"][self._series[m]["exclusions"][0]][:, 1],
            )
            ax.plot(
                self._series[0]["nodes"][self._series[0]["exclusions"][0]][:, 0],
                self._series[0]["nodes"][self._series[0]["exclusions"][0]][:, 1],
            )
            ax.scatter(
                self._coordinates[self._reference_index][0],
                self._coordinates[self._reference_index][1],
            )
            plt.show()
            raise ValueError("Particle outside of boundary.")
            del self_series
        return index

    def _local_coordinates(self, element_nodes):
        """
        Private method to find the particle position in terms of the local element
        coordinate system (zeta, eta, theta).

        Parameters
        ----------
        element_nodes : numpy.ndarray
            Nodal coordinate array.

        Returns
        -------
        zeta : float
            The local coordinate along the 1st axis. Value will be between 0.0-1.0.
        eta : float
            The local coordinate along the 2nd axis. Value will be between 0.0-1.0.
        theta : float
            The local coordinate along the 3rd axis. Value will be between 0.0-1.0.
        A : `numpy.ndarray`
            Element nodes coordinate array.

        """
        # Local coordinates
        A = np.ones((3, 4))
        A[1:, 0] = self._coordinates[self._reference_index]
        A[1:, 1:] = element_nodes[:3, :2].transpose()
        zeta = np.linalg.det(A[:, [0, 2, 3]]) / np.linalg.det(A[:, [1, 2, 3]])
        eta = np.linalg.det(A[:, [0, 3, 1]]) / np.linalg.det(A[:, [1, 2, 3]])
        theta = 1 - zeta - eta

        return zeta, eta, theta, A

    def _shape_function(self, zeta, eta, theta):
        """
        Private method to create the shape function array and it's derivatives
        up to 2nd order.

        Parameters
        ----------
        zeta : float
            The local coordinate along the 1st axis. Value will be between 0.0-1.0.
        eta : float
            The local coordinate along the 2nd axis. Value will be between 0.0-1.0.
        theta : float
            The local coordinate along the 3rd axis. Value will be between 0.0-1.0.

        Returns
        -------
        N : numpy.ndarray (6)
            Element shape function : [N1 N2 N3 N4 N5 N6]
        dN : numpy.ndarray (2,6)
            Element shape function 1st order derivatives :
            [[dN1/dzeta dN1/dzeta ...][dN1/deta dN1/deta ...]]
        d2N : numpy.ndarray(3,6)
            Element shape function 2nd order derivatives :
            [[d^2N1/dzeta^2 d^2N2/dzeta^2   ...]
             [d^2N1/dzeta^2 d^2N2/dzetadeta ...]
             [d^2N1/deta^2  d^2N2/deta^2    ...]]
        """
        if self._mesh_order == 1:
            N = np.asarray([zeta, eta, theta])
            dN = np.asarray([[1, 0, -1], [0, 1, -1]])
            d2N = None
        elif self._mesh_order == 2:
            N = np.asarray(
                [
                    zeta * (2 * zeta - 1),
                    eta * (2 * eta - 1),
                    theta * (2 * theta - 1),
                    4 * zeta * eta,
                    4 * eta * theta,
                    4 * theta * zeta,
                ]
            )
            dN = np.asarray(
                [
                    [
                        4 * zeta - 1,
                        0,
                        1 - 4 * theta,
                        4 * eta,
                        -4 * eta,
                        4 * (theta - zeta),
                    ],
                    [
                        0,
                        4 * eta - 1,
                        1 - 4 * theta,
                        4 * zeta,
                        4 * (theta - eta),
                        -4 * zeta,
                    ],
                ]
            )
            d2N = np.asarray(
                [
                    [4, 0, 4, 0, 0, -8],
                    [0, 0, 4, 4, -4, -4],
                    [0, 4, 4, 0, -8, 0],
                ]
            )

        return N, dN, d2N

    def _warp_increment(self, m, tri_idx):
        """
        Private method to calculate the warp increment between the reference and target
        image.

        Parameters
        ----------
        m : int
            Series index.
        tri_idx : int
            Element index.

        """

        self._warp_inc = np.zeros(6 * self._mesh_order)
        x = self._series[m]["nodes"][self._series[m]["elements"][tri_idx]]
        u = self._series[m]["results"]["displacements"][
            self._series[m]["elements"][tri_idx]
        ]

        # Get local coordinates.
        zeta, eta, theta, A = self._local_coordinates(x)

        # Shape function and derivatives.
        N, dN, d2N = self._shape_function(zeta, eta, theta)

        # Displacements
        self._warp_inc[:2] = N @ u

        # 1st Order Strains
        J_x_T = dN @ x
        J_u_T = dN @ u
        self._warp_inc[2:6] = (np.linalg.inv(J_x_T) @ J_u_T).flatten()

        # 2nd Order Strains
        if self._mesh_order == 2:
            K_u = d2N @ u
            dz = np.asarray(
                [
                    [x[1, 1] - x[2, 1], x[2, 1] - x[0, 1]],
                    [x[2, 0] - x[1, 0], x[0, 0] - x[2, 0]],
                ]
            )
            dz /= np.linalg.det(A[:, [1, 2, 3]])
            K_x_inv = np.asarray(
                [
                    [dz[0, 0] ** 2, 2 * dz[0, 0] * dz[0, 1], dz[0, 1] ** 2],
                    [
                        dz[0, 0] * dz[1, 0],
                        dz[0, 0] * dz[1, 1] + dz[0, 1] * dz[1, 0],
                        dz[0, 1] * dz[1, 1],
                    ],
                    [dz[1, 0] ** 2, 2 * dz[1, 0] * dz[1, 1], dz[1, 1] ** 2],
                ]
            )

            self._warp_inc[6:] = (K_x_inv @ K_u).flatten()

    def _check_update(self, m):
        if int(
            re.findall(
                r"\d+",
                self._series[self._reference_index]["images"]["f_img"],
            )[-1]
        ) != int(re.findall(r"\d+", self._series[m]["images"]["f_img"])[-1]):
            self._reference_index = m
            self._reference_update_register.append(m)

    def _strain_path(self):
        self._reference_update_register = []
        self._incs = np.zeros((len(self._series) + 1, 6 * self._mesh_order))
        for m in range(np.shape(self._series)[0]):
            self._check_update(m)
            tri_idx = self._element_locator(m)
            self._warp_increment(m, tri_idx)
            self._incs[m + 1] = self._warp_inc
            self._coordinates[m + 1] = self._coordinates[
                self._reference_index
            ] + self._warp_inc[:2] * int(self._track)
            self._warps[m + 1, :2] = (
                self._warps[self._reference_index, :2] + self._warp_inc[:2]
            )
            self._warps[m + 1, 2] = self._warps[self._reference_index, 2] + np.log(
                1 + self._warp_inc[2]
            )
            self._warps[m + 1, 3] = (
                self._warps[self._reference_index, 3] + self._warp_inc[3]
            )
            self._warps[m + 1, 4] = (
                self._warps[self._reference_index, 4] + self._warp_inc[4]
            )
            self._warps[m + 1, 5] = self._warps[self._reference_index, 5] + np.log(
                1 + self._warp_inc[5]
            )
            self._warps[m + 1, 6:] = (
                self._warps[self._reference_index, 6:] + self._warp_inc[6:]
            )
            self._volumes[m + 1] = self._volumes[self._reference_index] * (
                (1 + self._warp_inc[2]) * (1 + self._warp_inc[5])
                - self._warp_inc[3] * self._warp_inc[4]
            )

        self._strain_def()
        return True

    def _stress_path(self, model, state, parameters):
        """Under construction"""

        if model == "LinearElastic":
            # Put input checks here!
            model = geomat.models.LinearElastic(
                parameters=parameters,
                state=state,
                # log_severity = "verbose"
            )
        if model == "MCC":
            # Put input checks here!
            model = geomat.models.MCC(
                parameters=parameters, state=state, log_severity="verbose"
            )
        elif model == "SMCC":
            # Put input checks here!
            model = geomat.models.SMCC(
                parameters=parameters,
                state=state,  # log_severity="verbose"
            )
        strain_incs = np.diff(self._strains, axis=0)
        model.set_sigma_prime_tilde(self._stresses[0].T)
        model.set_Delta_epsilon_tilde(strain_incs[0])
        self._ps[0] = model.p_prime
        self._qs[0] = model.q
        self._states[0] = model.state
        for i in range(np.shape(self._series)[0]):
            try:
                model.set_sigma_prime_tilde(self._stresses[i].T)
                model.set_Delta_epsilon_tilde(-1 * strain_incs[i])
                model.solve()
            except Exception:
                log.error("geomat error. Stress path curtailed.")
                raise ValueError("geomat error. Stress path curtailed.")
            self._ps[i + 1] = model.p_prime
            self._qs[i + 1] = model.q
            self._states[i + 1] = model.state
            self._stresses[i + 1] = model.sigma_prime_tilde

        self._works[1:] = (
            np.sum(self._stresses[1:] * strain_incs, axis=-1) * self._volumes[1:]
        )

        return True

    def _strain_def(self):
        self._strains[:, 0] = self._warps[:, 2]
        self._strains[:, 1] = self._warps[:, 5]
        self._strains[:, 5] = self._warps[:, 3] + self._warps[:, 4]


class ParticleResults(ParticleBase):
    """
    ParticleResults class for geopyv.

    Parameters
    ----------
    data : dict
        geopyv data dict from Particle object.

    Attributes
    ----------
    data : dict
        geopyv data dict from Particle object.
    """

    def __init__(self, data):
        """Initialisation of geopyv ParticleResults class."""
        self.data = data
