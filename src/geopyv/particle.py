"""

Particle module for geopyv.

"""
import logging
import numpy as np
import geopyv as gp
from geopyv.object import Object
import re
import matplotlib.path as path

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
        if self.data["solved"] is not True or "results" not in self.data:
            log.error(
                "Particle not yet solved therefore no convergence data to plot. "
                "First, run :meth:`~geopyv.particle.Particle.solve()` to solve."
            )
            raise ValueError(
                "Particle not yet solved therefore no convergence data to plot. "
                "First, run :meth:`~geopyv.particle.Particle.solve()` to solve."
            )

        # Check input.
        if type(quantity) != str and quantity is not None:
            log.error(
                (
                    "`quantity` keyword argument type invalid. "
                    "Expected a `str` or `NoneType`, but got a {type3}."
                ).format(type3=type(quantity).__name__)
            )
            raise TypeError(
                (
                    "`quantity` keyword argument type invalid. "
                    "Expected a `str` or `NoneType`, but got a {type3}."
                ).format(type3=type(quantity).__name__)
            )
        elif quantity not in [
            "coordinates",
            "warps",
            "volumes",
            "stresses",
        ]:
            log.error(
                (
                    "`quantity` keyword argument value invalid. "
                    "Expected `coordinates`, `warps`, `volumes`, `stresses`, "
                    "but got {value}."
                ).format(value=quantity)
            )
            raise ValueError(
                (
                    "`quantity` keyword argument value invalid. "
                    "Expected `coordinates`, `warps`, `volumes`, `stresses`, "
                    "but got {value}."
                ).format(value=quantity)
            )
        if type(component) != int:
            log.error(
                (
                    "`component` keyword argument type invalid. "
                    "Expected a `int`, but got a {type3}."
                ).format(type3=type(component).__name__)
            )
            raise TypeError(
                (
                    "`component` keyword argument type invalid. "
                    "Expected a `int`, but got a {type3}."
                ).format(type3=type(component).__name__)
            )
        if component >= np.shape(self.data["results"][quantity])[1]:
            log.error(
                (
                    "`component` {input_value} is out of bounds for "
                    "axis 0 with size {max_value}."
                ).format(
                    max_value=np.shape(self.data["results"][quantity])[1] - 1,
                    input_value=component,
                )
            )
            raise IndexError(
                (
                    "`component` {input_value} is out of bounds for "
                    "axis 0 with size {max_value}."
                ).format(
                    max_value=np.shape(self.data["results"][quantity])[1] - 1,
                    input_value=component,
                )
            )
        if type(imshow) != bool:
            log.error(
                (
                    "`imshow` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(imshow).__name__)
            )
            raise TypeError(
                (
                    "`imshow` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(imshow).__name__)
            )
        if type(colorbar) != bool:
            log.error(
                (
                    "`colorbar` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(colorbar).__name__)
            )
            raise TypeError(
                (
                    "`colorbar` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(colorbar).__name__)
            )
        if isinstance(ticks, (tuple, list, np.ndarray)) is False and ticks is not None:
            log.error(
                (
                    "`ticks` keyword argument type invalid. "
                    "Expected a `tuple`, `list` or `numpy.ndarray`, "
                    "but got a `{type2}`."
                ).format(type2=type(ticks).__name__)
            )
            raise TypeError(
                (
                    "`ticks` keyword argument type invalid. "
                    "Expected a `tuple`, `list` or `numpy.ndarray`, "
                    "but got a `{type2}`."
                ).format(type2=type(ticks).__name__)
            )
        if type(alpha) != float:
            log.warning(
                (
                    "`alpha` keyword argument type invalid. "
                    "Expected a `float`, but got a `{type2}`.\n"
                    "Attempting conversion..."
                ).format(type2=type(alpha).__name__)
            )
            try:
                alpha = float(alpha)
                log.warning(
                    (
                        "`alpha` keyword argument type conversion successful. "
                        "New value: {value}"
                    ).format(value=alpha)
                )
            except ValueError:
                log.error(
                    "`alpha` keyword argument type conversion failed. "
                    "Input a `float`, 0.0-1.0."
                )
                raise TypeError(
                    "`alpha` keyword argument type conversion failed. "
                    "Input a `float`, 0.0-1.0."
                )
        elif alpha < 0.0 or alpha > 1.0:
            log.error(
                (
                    "`alpha` keyword argument value {value} out of range 0.0-1.0. "
                    "Input a `float`, 0.0-1.0."
                ).format(value=alpha)
            )
            raise ValueError(
                (
                    "`alpha` keyword argument value {value} out of range 0.0-1.0. "
                    "Input a `float`, 0.0-1.0."
                ).format(value=alpha)
            )
        if type(axis) != bool:
            log.error(
                (
                    "`axis` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(axis).__name__)
            )
            raise TypeError(
                (
                    "`axis` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(axis).__name__)
            )
        if xlim is not None:
            if isinstance(xlim, (tuple, list, np.ndarray)) is False:
                log.error(
                    (
                        "`xlim` keyword argument type invalid. "
                        "Expected a `tuple`, `list`, `numpy.ndarray` or `NoneType`, "
                        "but got a {type5}."
                    ).format(type5=type(xlim).__name__)
                )
                raise TypeError(
                    (
                        "`xlim` keyword argument type invalid. "
                        "Expected a `tuple`, `list`, `numpy.ndarray` or `NoneType`, "
                        "but got a {type5}."
                    ).format(type5=type(xlim).__name__)
                )
            elif np.shape(xlim)[0] != 2:
                log.error(
                    (
                        "`xlim` keyword argument primary axis size invalid. "
                        "Expected 2, but got {size}."
                    ).format(size=np.shape(xlim)[0])
                )
                raise ValueError(
                    (
                        "`xlim` keyword argument primary axis size invalid. "
                        "Expected 2, but got {size}."
                    ).format(size=np.shape(xlim)[0])
                )
        if ylim is not None:
            if isinstance(ylim, (tuple, list, np.ndarray)) is False:
                log.error(
                    (
                        "`ylim` keyword argument type invalid. "
                        "Expected a `tuple`, `list`, `numpy.ndarray` or `NoneType`, "
                        "but got a {type5}."
                    ).format(type5=type(ylim).__name__)
                )
                raise TypeError(
                    (
                        "`ylim` keyword argument type invalid. "
                        "Expected a `tuple`, `list`, `numpy.ndarray` or `NoneType`, "
                        "but got a {type5}."
                    ).format(type5=type(ylim).__name__)
                )
            elif np.shape(ylim)[0] != 2:
                log.error(
                    (
                        "`ylim` keyword argument primary axis size invalid. "
                        "Expected 2, but got {size}."
                    ).format(size=np.shape(ylim)[0])
                )
                raise ValueError(
                    (
                        "`ylim` keyword argument primary axis size invalid. "
                        "Expected 2, but got {size}."
                    ).format(size=np.shape(ylim)[0])
                )
        if type(show) != bool:
            log.error(
                (
                    "`show` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(show).__name__)
            )
            raise TypeError(
                (
                    "`show` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(show).__name__)
            )
        if type(block) != bool:
            log.error(
                (
                    "`block` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(block).__name__)
            )
            raise TypeError(
                (
                    "`block` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(block).__name__)
            )
        if type(save) != str and save is not None:
            log.error(
                (
                    "`save` keyword argument type invalid. "
                    "Expected a `str` or `NoneType`, but got a `{type3}`."
                ).format(type3=type(save).__name__)
            )
            raise TypeError(
                (
                    "`save` keyword argument type invalid. "
                    "Expected a `str` or `NoneType`, but got a `{type3}`."
                ).format(type3=type(save).__name__)
            )

        fig, ax = gp.plots.trace_particle(
            data=self.data,
            quantity=quantity,
            component=component,
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
        volume_0=1.0,
        moving=True,
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
        moving : bool
            Boolean for Lagrangian (False) or Eulerian (True) specification.
            Defaults to False.
        """

        self._initialised = False
        # Check types.
        if series.data["type"] != "Sequence" and series.data["type"] != "Mesh":
            log.error(
                "Invalid series type. Must be gp.sequence.Sequence or gp.mesh.Mesh."
            )
        if type(coordinate_0) != np.ndarray:
            log.error("Invalid coordinate type. Must be numpy.ndarray.")
        elif np.shape(coordinate_0) != np.shape(np.empty(2)):
            log.error("Invalid coordinate shape. Must be (2).")
        if type(warp_0) != np.ndarray:
            log.error("Invalid initial warp type. Must be numpy.ndarray.")
        elif np.shape(warp_0) != np.shape(np.empty(12)):
            log.error("Invalid initial warp shape. Must be (12).")
        # if type(volume_0) != float or type(volume_0)!= np.float64:
        #    log.error("Invalid initial volume type. Must be a float.")
        if type(moving) != bool:
            log.error("Invalid moving type. Must be a bool.")

        if series.data["type"] == "Sequence":
            self._series_type = "Sequence"
            if series.data["file_settings"]["save_by_reference"]:
                self._series = np.empty(
                    np.shape(series.data["meshes"])[0], dtype=object
                )
                for i in range(np.shape(series.data["meshes"])[0]):
                    self._series[i] = gp.io.load(
                        filename=series.data["file_settings"]["mesh_folder"]
                        + series.data["meshes"][i]
                    ).data
            else:
                self._series = series.data["meshes"]
        else:
            self._series_type = "Mesh"
            self._series = np.asarray([series.data])
        self._moving = moving

        if self._series[0]["mask"][int(coordinate_0[1]), int(coordinate_0[0])] == 0:
            log.warning(
                "`coordinate_0` keyword argument value out of boundary.\n"
                "Select `coordinate_0`..."
            )
            coordinate_0 = gp.gui.selectors.coordinate.CoordinateSelector()

        self._coordinates = np.zeros((len(self._series) + 1, 2))
        self._warps = np.zeros((len(self._series) + 1, 12))
        self._volumes = np.zeros(len(self._series) + 1)
        self._stresses = np.zeros((len(self._series) + 1, 6))

        self._coordinates[0] = coordinate_0
        self._warps[0] = warp_0
        self._volumes[0] = volume_0

        self._reference_index = 0
        self.solved = False

        self._initialised = True
        self.data = {
            "type": "Particle",
            "solved": self.solved,
            "series_type": self._series_type,
            "moving": self._moving,
            "coordinate_0": self._coordinates[0],
            "warp_0": self._warps[0],
            "volume_0": self._volumes[0],
            "image_0": self._series[0]["images"]["f_img"],
        }
        # "series": self._series,

    def solve(self):  # model, statev, props):
        """

        Method to calculate the strain path of the particle from the
        mesh sequence and optionally the stress path employing the
        model specified by the input parameters.

        """

        self.solved += self._strain_path()
        # self.solved += self._stress_path(model, statev, props)
        self._results = {
            "coordinates": self._coordinates,
            "warps": self._warps,
            "volumes": self._volumes,
            "stresses": self._stresses,
        }
        self.data.update({"results": self._results})
        self.data["solved"] = bool(self.solved)
        return self.solved

    def _triangulation_locator(self, m):
        """

        Method to locate the numerical particle within the mesh,
        returning the current element_nodes index.


        Parameters
        ----------
        mesh : `geopyv.Mesh` object
            The relevant mesh.

        """

        diff = (
            self._series[m]["nodes"] - self._coordinates[self._reference_index]
        )  # Particle-mesh node positional vector.
        dist = np.einsum(
            "ij,ij->i", diff, diff
        )  # Particle-mesh node "distances" (truly distance^2).
        tri_idxs = np.argwhere(
            np.any(self._series[m]["elements"] == np.argmin(dist), axis=1)
        ).flatten()  # Retrieve relevant element_nodes indices.
        for i in range(len(tri_idxs)):
            if path.Path(
                self._series[m]["nodes"][self._series[m]["elements"][tri_idxs[i]]]
            ).contains_point(
                self._coordinates[self._reference_index]
            ):  # Check if the element_nodes includes the particle coordinates.
                break  # If the correct element_nodes is identified, stop the search.

        return tri_idxs[i]  # Return the element_nodes index.

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
        self._warp_inc = np.zeros(12)
        element_nodes = self._series[m]["nodes"][self._series[m]["elements"][tri_idx]]
        displacements = self._series[m]["results"]["displacements"][
            self._series[m]["elements"][tri_idx]
        ]

        # Get local coordinates.
        zeta, eta, theta, A = self._local_coordinates(element_nodes)

        # Shape function and derivatives.
        N, dN, d2N = self._shape_function(zeta, eta, theta)

        # Displacements
        self._warp_inc[:2] = N @ displacements

        # 1st Order Strains
        J_x_T = dN @ element_nodes
        J_u_T = dN @ displacements
        self._warp_inc[2:6] = (np.linalg.inv(J_x_T) @ J_u_T).flatten()

        # 2nd Order Strains
        K_u = d2N @ displacements
        J_zeta = np.zeros((2, 2))
        J_zeta[0, 0] = element_nodes[1, 1] - element_nodes[2, 1]
        J_zeta[0, 1] = element_nodes[2, 0] - element_nodes[1, 0]
        J_zeta[1, 0] = element_nodes[2, 1] - element_nodes[0, 1]
        J_zeta[1, 1] = element_nodes[0, 0] - element_nodes[2, 0]
        J_zeta /= np.linalg.det(A[:, [1, 2, 3]])

        K_x_inv = np.zeros((3, 3))
        K_x_inv[0, 0] = J_zeta[0, 0] ** 2
        K_x_inv[0, 1] = 2 * J_zeta[0, 0] * J_zeta[0, 1]
        K_x_inv[0, 2] = J_zeta[0, 1] ** 2
        K_x_inv[1, 0] = J_zeta[0, 0] * J_zeta[1, 0]
        K_x_inv[1, 1] = J_zeta[0, 0] * J_zeta[1, 1] + J_zeta[1, 0] * J_zeta[0, 1]
        K_x_inv[1, 2] = J_zeta[0, 1] * J_zeta[1, 1]
        K_x_inv[2, 0] = J_zeta[1, 0] ** 2
        K_x_inv[2, 1] = 2 * J_zeta[1, 0] * J_zeta[1, 1]
        K_x_inv[2, 2] = J_zeta[1, 1] ** 2

        self._warp_inc[6] = (
            K_u[0, 0] * J_zeta[0, 0] ** 2
            + 2 * K_u[1, 0] * J_zeta[0, 0] * J_zeta[1, 0]
            + K_u[2, 0] * J_zeta[1, 0] ** 2
        )
        self._warp_inc[7] = (
            K_u[0, 1] * J_zeta[0, 0] ** 2
            + 2 * K_u[1, 1] * J_zeta[0, 0] * J_zeta[1, 0]
            + K_u[2, 1] * J_zeta[1, 0] ** 2
        )
        self._warp_inc[8] = (
            K_u[0, 0] * J_zeta[0, 0] * J_zeta[0, 1]
            + K_u[1, 0] * (J_zeta[0, 0] * J_zeta[1, 1] + J_zeta[1, 0] * J_zeta[0, 1])
            + K_u[2, 0] * J_zeta[1, 0] * J_zeta[1, 1]
        )
        self._warp_inc[9] = (
            K_u[0, 1] * J_zeta[0, 0] * J_zeta[0, 1]
            + K_u[1, 1] * (J_zeta[0, 0] * J_zeta[1, 1] + J_zeta[1, 0] * J_zeta[0, 1])
            + K_u[2, 1] * J_zeta[1, 0] * J_zeta[1, 1]
        )
        self._warp_inc[10] = (
            K_u[0, 0] * J_zeta[0, 1] ** 2
            + 2 * K_u[1, 0] * J_zeta[0, 1] * J_zeta[1, 1]
            + K_u[2, 0] * J_zeta[1, 1] ** 2
        )
        self._warp_inc[11] = (
            K_u[0, 1] * J_zeta[0, 1] ** 2
            + 2 * K_u[1, 1] * J_zeta[0, 1] * J_zeta[1, 1]
            + K_u[2, 1] * J_zeta[1, 1] ** 2
        )

    def _strain_path(self):
        """Method to calculate and store stress path data for the particle object."""
        for m in range(len(self._series)):
            if int(
                re.findall(
                    r"\d+",
                    self._series[self._reference_index]["images"]["f_img"],
                )[-1]
            ) != int(re.findall(r"\d+", self._series[m]["images"]["f_img"])[-1]):
                self._reference_index = m
            tri_idx = self._triangulation_locator(
                m
            )  # Identify the relevant element of the mesh.
            self._warp_increment(m, tri_idx)  # Calculate the nodal weightings.
            self._coordinates[m + 1] = self._coordinates[
                self._reference_index
            ] + self._warp_inc[:2] * int(
                self._moving
            )  # Update the particle positional coordinate.
            # i.e. (reference + mesh interpolation).
            self._warps[m + 1] = self._warps[self._reference_index] + self._warp_inc
            self._volumes[m + 1] = (
                self._volumes[m]
                * (1 + (self._warps[m + 1, 2] - self._warps[m, 2]))
                * (1 + (self._warps[m + 1, 5] - self._warps[m, 5]))
            )
            # Update the particle volume.
            # i.e. (reference*(1 + volume altering strain components)).
        return True


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
        """Initialisation of geopyv SequenceResults class."""
        self.data = data
