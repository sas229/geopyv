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
from geomat.abstract import Elastoplastic  # noqa: F401
from geomat.utilities import Derivatives  # noqa: F401
from geomat.models import LinearElastic, MCC, SMCC, C2MC, EMC  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401
import traceback

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
        coordinate=np.zeros(2),
        warp=np.zeros(12),
        stress=np.zeros(6),
        volume=1.0,
        track=True,
        field=False,
        ID="",
    ):
        """Initialisation of geopyv particle object.

        Parameters
        ----------
        meshes : `numpy.ndarray` of geopyv.mesh.Mesh objects
            Sequence for the particle object to track.
        coordinate : numpy.ndarray (2)
            Initial particle coordinate (x,y)
        p_init : `numpy.ndarray` (12), optional
            Initial warp vector.
        vol : float
            Volume represented by the particle.
        track : bool
            Boolean for Lagrangian (False) or Eulerian (True) specification.
            Defaults to False.

        Note ::
        If the series supplied has been calibrated, it is assumed
        that the coordinate provided is in object space. Else, in image space.
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

        check = gp.check._check_type(warp, "warp", [np.ndarray])
        if check:
            try:
                warp = np.asarray(warp)
                self._report(gp.check._conversion(warp, "warp", np.ndarray), "Warning")
            except Exception:
                self._report(check, "TypeError")
        self._report(gp.check._check_dim(warp, "warp", 1), "ValueError")
        self._report(gp.check._check_axis(warp, "warp", 0, [6, 12]), "ValueError")
        check = gp.check._check_type(
            volume, "volume", [float, np.floating, np.float64, np.float32]
        )
        if check:
            try:
                volume = float(volume)
                self._report(gp.check._conversion(volume, "volume", float), "Warning")
            except Exception:
                self._report(check, "TypeError")
        self._report(gp.check._check_range(volume, "volume", 0.0), "ValueError")
        self._report(gp.check._check_type(track, "track", [bool]), "TypeError")

        self._series = series
        if self._series.data["type"] == "Sequence":
            self._inc_no = len(self._series.data["meshes"]) + 1
        else:
            self._inc_no = 2
        self._track = track
        self._field = field
        self._update_mesh(0)
        self._mesh_order = self._cm.data["mesh_order"]
        if self._field is False:
            self._rm = self._cm

        if self._report(
            gp.check._check_type(coordinate, "coordinate", [np.ndarray]), "Warning"
        ):
            selector = gp.gui.selectors.coordinate.CoordinateSelector()
            f_img = gp.image.Image(filepath=self._series[0]["images"]["f_img"])
            coordinate = selector.select(f_img, gp.templates.Circle(20))
        elif self._report(gp.check._check_dim(coordinate, "coordinate", 1), "Warning"):
            selector = gp.gui.selectors.coordinate.CoordinateSelector()
            f_img = gp.image.Image(filepath=self._series[0]["images"]["f_img"])
            coordinate = selector.select(f_img, gp.templates.Circle(20))
        elif self._report(
            gp.check._check_axis(coordinate, "coordinate", 0, [2]), "Warning"
        ):
            selector = gp.gui.selectors.coordinate.CoordinateSelector()
            f_img = gp.image.Image(filepath=self._series[0]["images"]["f_img"])
            coordinate = selector.select(f_img, gp.templates.Circle(20))
        check = gp.check._check_type(ID, "ID", [str])
        if check:
            try:
                ID = str(ID)
            except Exception:
                self._report(check, "TypeError")

        # Improve by checking boundary itself so image and object space are covered.
        # if series.data["calibrated"] is False:
        #     if self._series[0]["mask"][int(coordinate[1]), int(coordinate[0])] == 0:
        #         log.warning(
        #             "`coordinate` keyword argument value out of boundary.\n"
        #             "Select `coordinate`..."
        #         )
        #         selector = gp.gui.selectors.coordinate.CoordinateSelector()
        #         f_img = gp.image.Image(filepath=self._series[0]["images"]["f_img"])
        #         coordinate = selector.select(f_img, gp.templates.Circle(20))

        self._reference_index = 0
        self.solved = False
        self._calibrated = series.data["calibrated"]
        self._reference_update_register = []
        self._incs = np.zeros((self._inc_no, 6 * self._mesh_order))
        self._coordinates = np.zeros((self._inc_no, 2))
        self._warps = np.zeros((self._inc_no, 6 * self._mesh_order))
        self._volumes = np.zeros(self._inc_no)
        self._stresses = np.zeros((self._inc_no, 6))
        self._works = np.zeros(self._inc_no)
        self._strains = np.zeros((self._inc_no, 6))
        self._ID = ID

        self._warps[0] = warp[: np.shape(self._warps)[1]]
        self._volumes[0] = volume
        self._stresses[0] = stress
        self._coordinates[0] = coordinate
        self._adrift = False
        if self._calibrated is True:
            self._nref = "Nodes"
            self._cref = "Centroids"
            self._dref = "Displacements"
            if self._series.data["type"] == "Sequence":
                self._plotting_coordinates = np.zeros((self._inc_no, 2))
            else:
                self._plotting_coordinates = np.zeros((2, 2))
            self._plotting_coordinates[0] = self._coord_map()
        else:
            self._nref = "nodes"
            self._cref = "centroids"
            self._dref = "displacements"

        self._initialised = True
        self.data = {
            "type": "Particle",
            "ID": self._ID,
            "solved": self.solved,
            "calibrated": self._calibrated,
            # "series_type": self._series_type,
            "image_0": self._cm.data["images"]["f_img"],
            "track": self._track,
        }
        if self._field:
            del self._cm

    def _update_mesh(self, index):
        try:
            del self._cm
        except Exception:
            pass
        if self._series.data["type"] == "Sequence":
            if self._series.data["file_settings"]["save_by_reference"] is True:
                self._cm = self._series._load_mesh(index, obj=True, verbose=False)
            else:
                self._cm = self._series.data["meshes"][index]
        else:
            self._cm = self._series

    def _coord_map(self):
        """

        Private method to map from image to object space (or vice versa)
        from the user-given coordinate input. Note, this applies quadratic mapping
        for a 2nd order mesh and linear mapping for a 1st order mesh.

        """

        tri_idx = self._element_locator()

        x = self._cm.data["nodes"][self._cm.data["elements"][tri_idx]]
        X = self._cm.data["Nodes"][self._cm.data["elements"][tri_idx]]
        zeta, eta, theta, A = self._local_coordinates(X)
        N, dN, d2N = self._shape_function(zeta, eta, theta)
        return N @ x

    def solve(self, *, model=None, state=None, parameters=None, intype=0, verbose=True):
        """

        Method to solve for the particle.

        Parameters
        ----------
        model : str, optional
            If specified, determines the constitutive model to be used
            from the geomat package. Options are:
            - LinearElastic
            - MCC
            - SMCC
        state : np.ndarray, optional
            If a model is specified, the corresponding initial state
            variables must be specified. Formatting is as follows:
            - LinearElastic : np.array([])
        parameters : np.ndarray, optional
            If a model is specified, the corresponding model parameters
            must be specified. Formatting is as follows:
            - LinearElastic
            - MCC
            - SMCC
        intype : int, optional
            Base interpolation order. Options are:
            0 (default) : strains are interpolated from nodal displacementss
                i.e. strains from a 1st order mesh are singular per element.
            1 : strains are interpolated from nodal strains i.e. strains from
            a 1st order mesh are linearly interpolatable.
        """
        # Checks.

        # Store variables.
        self._intype = intype

        # Solving.
        if self._field is False:
            self.solved += self._strain_path_full(verbose)
        self._strain_def()
        if model:
            self.solved += self._stress_path(model, state, parameters)
            self._results = {
                "coordinates": self._coordinates,
                "warps": self._warps,
                "volumes": self._volumes,
                "stresses": self._stresses,
                "strains": self._strains,
                "mean_effective_stresses": self._ps,
                "deviatoric_stresses": self._qs,
                "states": self._states,
                "works": self._works,
            }
        else:
            self._results = {
                "coordinates": self._coordinates,
                "warps": self._warps,
                "strains": self._strains,
                "volumes": self._volumes,
            }
        if self._calibrated is True:
            self.data["plotting_coordinates"] = self._plotting_coordinates
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

    def _element_locator(self):
        """

        Private method to identify the current element index
        for the particle.

        Parameters
        ----------
        m : int
            Series index.

        """
        elements = self._cm.data["elements"]
        nodes = self._cm.data[self._nref]
        centroids = self._cm.data[self._cref]
        flag = False
        cdiff = centroids - self._coordinates[self._reference_index]
        cdist = np.einsum("ij,ij->i", cdiff, cdiff)
        cdist_sorted = np.argpartition(cdist, min(10, math.ceil(0.05 * len(cdist))))
        for index in cdist_sorted:
            hull = sp.spatial.Delaunay(nodes[elements[index]][:3])
            if hull.find_simplex(self._coordinates[self._reference_index]) >= 0:
                flag = True
                if self._adrift is True:
                    self._adrift = False
                    log.warning(
                        (
                            "{} returned to shore at {}. "
                            "Returning to regular interpolation."
                        ).format(
                            "Particle " + self._ID,
                            np.round(self._coordinates[self._reference_index], 2),
                        )
                    )
                break
        if flag is False:
            if self._adrift is False:
                self._adrift = True
                log.warning(
                    (
                        "{} adrift at {}. "
                        "Proceeding with nearest element extrapolation."
                    ).format(
                        "Particle " + self._ID,
                        np.round(self._coordinates[self._reference_index], 2),
                    )
                )
            try:
                ndiff = nodes - self._coordinates[self._reference_index]
                ndist = np.einsum("ij,ij->i", ndiff, ndiff)
                ndist_sorted = np.argpartition(ndist, 2)
                index = np.argwhere(
                    np.any(elements == ndist_sorted[0], axis=1)
                    * np.any(elements == ndist_sorted[1], axis=1)
                )[0][0]
            except Exception:
                fig, ax = plt.subplots()
                ax.plot(
                    nodes[self._cm.data["exclusions"][0], 0],
                    nodes[self._cm.data["exclusions"][0], 1],
                    color="purple",
                )
                ax.plot(
                    nodes[self._cm.data["exclusions"][0][[0, -1]], 0],
                    nodes[self._cm.data["exclusions"][0][[0, -1]], 1],
                    color="purple",
                )
                ax.scatter(
                    nodes[elements[cdist_sorted[:10]], 0],
                    nodes[elements[cdist_sorted[:10]], 1],
                    color="b",
                )
                ax.scatter(
                    self._coordinates[self._reference_index, 0],
                    self._coordinates[self._reference_index, 1],
                    color="r",
                )
                for i in cdist_sorted[:10]:
                    ax.plot(
                        nodes[elements[i, [0, 1, 2, 0]], 0],
                        nodes[elements[i, [0, 1, 2, 0]], 1],
                        color="orange",
                    )
                # ax.plot(
                #     nodes[elements[index,[0,1,2,0]], 0],
                #     nodes[elements[index,[0,1,2,0]], 1],
                #     color = "g"
                # )
                ax.plot(
                    self._coordinates[: self._reference_index + 1, 0],
                    self._coordinates[: self._reference_index + 1, 1],
                    color="g",
                )
                print(np.round(nodes[ndist_sorted[:10]], 2))
                print(ndist[ndist_sorted[:10]])
                ax.set_aspect("equal", "box")
                plt.show()
                input()

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
        else:
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
        x = self._cm.data[self._nref][self._cm.data["elements"][tri_idx]]
        u = self._cm.data["results"][self._dref][self._cm.data["elements"][tri_idx]]

        # Get local coordinates.
        zeta, eta, theta, A = self._local_coordinates(x)

        # Shape function and derivatives.
        N, dN, d2N = self._shape_function(zeta, eta, theta)

        # Plotting coordinates.
        if self._calibrated is True:
            self._plotting_coordinates[m + 1] = (
                N @ self._cm.data["nodes"][self._cm.data["elements"][tri_idx]]
            )

        # Displacements
        self._warp_inc[:2] = N @ u

        # 1st Order Strains
        if self._intype == 0:
            J_x_T = dN @ x
            J_u_T = dN @ u
            self._warp_inc[2:6] = (np.linalg.inv(J_x_T) @ J_u_T).flatten()
        elif self._intype == 1:
            u_prime = np.zeros((3 * self._mesh_order, 4))
            for i in range(3 * self._mesh_order):
                u_prime[i] = self._cm.data["results"]["subsets"][
                    self._cm.data["elements"][tri_idx][i]
                ]["results"]["p"][2:6].flatten()
            self._warp_inc[2:6] = N @ u_prime

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
                self._rm.data["images"]["f_img"],
            )[-1]
        ) != int(re.findall(r"\d+", self._cm.data["images"]["f_img"])[-1]):
            self._reference_index = m
            self._reference_update_register.append(m)
            del self._rm
            self._rm = self._series._load_mesh(m, obj=True, verbose=False)

    def _strain_path_full(self, verbose):
        with alive_bar(
            self._inc_no - 1,
            dual_line=True,
            bar="blocks",
            title="Solving particle ...",
            disable=(not verbose),
        ) as bar:
            for m in range(self._inc_no - 1):
                self._update_mesh(m)
                self._check_update(m)
                tri_idx = self._element_locator()
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
                bar()
        return True

    def _strain_path_inc(self, m, cm, rm, intype=0):
        self._intype = intype
        self._update_cm_rm(m, cm, rm)
        tri_idx = self._element_locator()
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

        return True

    def _update_cm_rm(self, m, cm, rm):
        try:
            if self._rm != rm:
                self._reference_index = m
                self._reference_update_register.append(m)
            del self._cm
            del self._rm
        except Exception:
            pass
        self._cm = cm
        self._rm = rm

    def _stress_path(self, model, state, parameters):
        """Under construction"""
        self._ps = np.zeros(self._inc_no)
        self._qs = np.zeros(self._inc_no)
        self._states = np.zeros((self._inc_no, len(state)))
        if model == "LinearElastic":
            # Put input checks here!
            model = LinearElastic(
                parameters=parameters,
                state=state,
                # log_severity = "verbose"
            )
        if model == "MCC":
            # Put input checks here!
            model = MCC(
                parameters=parameters,
                state=state,
                # log_severity="verbose"
            )

        elif model == "SMCC":
            # Put input checks here!
            model = SMCC(
                parameters=parameters,
                state=state,
                # log_severity="verbose"
            )
        strain_incs = np.diff(self._strains, axis=0)
        model.set_sigma_prime_tilde(self._stresses[0].T)
        model.set_Delta_epsilon_tilde(-strain_incs[0])
        self._ps[0] = model.p_prime
        self._qs[0] = model.q
        # self._states[0] = model.state
        for i in range(self._inc_no - 1):
            try:
                model.set_sigma_prime_tilde(self._stresses[i].T)
                model.set_Delta_epsilon_tilde(strain_incs[i])
                model.solve()
            except Exception:
                print(
                    self._ID,
                    i,
                )
                print(strain_incs[i - 5 : i + 5])
                print(
                    np.round(self._coordinates[0], 2), np.round(self._coordinates[i], 2)
                )
                fig, (ax, bx) = plt.subplots(2, 1)
                ax.plot(range(i + 1), self._strains[: i + 1, 0], label=r"$du/dx$")
                ax.plot(range(i + 1), self._strains[: i + 1, 1], label=r"$dv/dy$")
                ax.plot(range(i + 1), self._strains[: i + 1, 5], label=r"$\gamma$")
                bx.plot(
                    range(i + 1), self._stresses[: i + 1, 0], label=r"$\sigma_{xx}$"
                )
                bx.plot(
                    range(i + 1), self._stresses[: i + 1, 1], label=r"$\sigma_{yy}$"
                )
                bx.plot(
                    range(i + 1), self._stresses[: i + 1, 2], label=r"$\sigma_{zz}$"
                )
                bx.plot(
                    range(i + 1), self._stresses[: i + 1, 5], label=r"$\sigma_{xy}$"
                )
                ax.legend()
                bx.legend()
                plt.show()

                print(traceback.format_exc())
                log.error("geomat error. Stress path curtailed.")
                raise ValueError("geomat error. Stress path curtailed.")
            # print(
            #     "Increment {}: p = {:.2f} ; q = {:.2f} ; q/p = {:.2f}".format(
            #         0, model.p_prime, model.q, model.q / model.p_prime
            #     )
            # )
            self._ps[i + 1] = model.p_prime
            self._qs[i + 1] = model.q
            # self._states[i + 1] = model.state
            self._stresses[i + 1] = model.sigma_prime_tilde

        self._works[1:] = (
            np.sum(self._stresses[1:] * strain_incs, axis=-1) * self._volumes[1:]
        )

        return True

    def _strain_def(self):
        self._strains[:, 5] = -(self._warps[:, 3] + self._warps[:, 4]) / 2
        # self._strains[:, 0] = (
        #     self._warps[:,2]*np.cos(0.5*self._strains[:,5])**2
        #     + self._warps[:,5]*np.sin(0.5*self._strains[:,5])**2
        #     + self._strains[:,5]*np.sin(0.5*self._strains[:,5])
        #     * np.cos(0.5*self._strains[:,5])
        # )
        # self._strains[:,1] = (
        #     self._warps[:,2]*np.cos(-0.5*self._strains[:,5])**2
        #     + self._warps[:,5]*np.sin(-0.5*self._strains[:,5])**2
        #     + self._strains[:,5]*np.sin(-0.5*self._strains[:,5])
        #     * np.cos(-0.5*self._strains[:,5])
        # )
        self._strains[:, 0] = self._warps[:, 2]
        self._strains[:, 1] = self._warps[:, 5]


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
