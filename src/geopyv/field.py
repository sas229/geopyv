"""

Field module for geopyv.

"""
import logging
import numpy as np
import geopyv as gp
from geopyv.object import Object
import gmsh
from scipy.optimize import minimize_scalar
from alive_progress import alive_bar
import re
import traceback

log = logging.getLogger(__name__)


class FieldBase(Object):
    """
    Field base class to be used as a mixin.
    """

    def __init__(self):
        super().__init__(object_type="Field")
        """

        Field base class initialiser.

        """

    def inspect(self, mesh=True, show=True, block=True, save=None):
        """
        Method to show the particles and associated representative areas.

        Parameters
        ----------
        mesh : bool, optional
            Control whether the mesh is plotted.
            Defaults to True.
        show : bool, optional
            Control whether the plot is displayed.
        block : bool, optional
            Control whether the plot blocks execution until closed.
        save : str, optional
            Name to use to save plot. Uses default extension of `.png`.
        """

        # Check inputs.
        self._report(gp.check._check_type(mesh, "mesh", [bool]), "TypeError")
        self._report(gp.check._check_type(show, "show", [bool]), "TypeError")
        self._report(gp.check._check_type(block, "block", [bool]), "TypeError")
        self._report(gp.check._check_type(save, "save", [str, type(None)]), "TypeError")

        log.info("Inspecting field...")
        fig, ax = gp.plots.inspect_field(
            self.data, mesh=mesh, show=show, block=block, save=save
        )
        return fig, ax

    def trace(
        self,
        *,
        quantity="warps",
        particle_index=None,
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
        self._report(gp.check._check_type(quantity, "quantity", [str]), "TypeError")
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
                component,
                "component",
                1,
                self.data["particles"][0].data["results"][quantity],
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

        if particle_index is None:
            log.info("Tracing field...")
            obj_type = "Field"
            data = self.data
        else:
            log.info("Tracing particle {}...".format(particle_index))
            obj_type = "Particle"
            data = self.data["particles"][particle_index].data
        fig, ax = gp.plots.trace_particle(
            data=data,
            quantity=quantity,
            component=component,
            obj_type=obj_type,
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

    def contour(
        self,
        *,
        quantity="warps",
        mesh_index=None,
        component=0,
        imshow=True,
        colorbar=True,
        ticks=None,
        alpha=0.75,
        levels=None,
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
        self._report(gp.check._check_type(quantity, "quantity", [str]), "TypeError")
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
                component,
                "component",
                1,
                self.data["particles"][0]["results"][quantity],
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
        types = [int, tuple, list, np.ndarray, type(None)]
        self._report(gp.check._check_type(levels, "levels", types), "TypeError")
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
        if mesh_index is None:
            mesh_index = self.data["number_images"] - 1
        log.info("Contour field...")
        data = self.data
        fig, ax = gp.plots.contour_field(
            data=data,
            mesh_index=mesh_index,
            quantity=quantity,
            component=component,
            imshow=imshow,
            colorbar=True,
            ticks=ticks,
            alpha=alpha,
            levels=levels,
            axis=axis,
            xlim=xlim,
            ylim=ylim,
            show=show,
            block=block,
            save=save,
        )
        return fig, ax

    def accumulation(
        self,
        *,
        quantity="R",
        window=None,
        imshow=True,
        colorbar=True,
        ticks=None,
        alpha=0.75,
        levels=None,
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
        self._report(gp.check._check_type(quantity, "quantity", [str]), "TypeError")
        if quantity:
            self._report(
                gp.check._check_value(
                    quantity,
                    "quantity",
                    ["u", "v", "u_x", "e_xy", "v_y", "R"],
                ),
                "ValueError",
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
        types = [int, tuple, list, np.ndarray, type(None)]
        self._report(gp.check._check_type(levels, "levels", types), "TypeError")
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

        fig, ax = gp.plots.accumulation_field(
            data=self.data,
            window=window,
            quantity=quantity,
            imshow=imshow,
            colorbar=True,
            ticks=ticks,
            alpha=alpha,
            levels=levels,
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
        particle_index,
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
                    ["coordinates", "warps", "volumes", "stresses", "works"],
                ),
                "ValueError",
            )
        self._report(
            gp.check._check_type(components, "components", [list, type(None)]),
            "TypeError",
        )
        # if components:
        #    for component in components:
        #        self._report(
        #            gp.check._check_index(
        #                component, "component", 1, self.data["results"][quantity]
        #            ),
        #            "IndexError",
        #        )
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
        log.info("Particle {} history...".format(particle_index))
        fig, ax = gp.plots.history_particle(
            data=self.data["particles"][particle_index].data,
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


class Field(FieldBase):
    def __init__(
        self,
        *,
        series=None,
        target_particles=1000,
        track=True,
        boundary=None,
        exclusions=[],
        coordinates=None,
        volumes=None,
        stresses=np.zeros(6),
        strains=np.zeros(6),
        ID="",
    ):
        """
        Initialisation of geopyv field object.

        Parameters
        ----------
        series : gp.sequence.Sequence object or gp.mesh.Mesh object
            The base series for field object interpolation.
        target_particles : int, optional
            Target number of particles. Defaults to a value of 1000.
        track : bool, optional
            Boolean to specify if particles should move or remain static.
            True: move (Lagrangian), False: static (Eularian). Defaults to True.
        boundary : numpy.ndarray (N,2), optional
            Array of coordinates to define the particle auto-distribution mesh
            boundary.
        exclusions : list, optional
            List to define the particle auto-distribution mesh exclusions.
            Shape of `numpy.ndarray` (N,2).
        coordinates : numpy.ndarray (N,2), optional
            Array of coordinates to define the initial particle positions.
        volumes : numpy.ndarray (N,), optional
            Array of volumes for particle representation.
            Defaults to np.ones(N) i.e. measure of volumetric strain.
        stresses : numpy.ndarray (N,6) or (2,6) or (6,), optional.
            Stress state specification. If:
            (N,6) - individual particle specification (where N is the number
                    of particles).
            (2,6) - linearly varying stress field with depth (1st row and 2nd
                    row vertically highest and lowest respectively, according
                    to the boundary definition.
            (6,)  - uniform stress field.
            Defaults to np.zeros(6).

        Note ::
        Two kwargs groups for particle distribution:
        if coordinates is not None:
            1. User-defined : coordinates, volumes.
        else:
            2. Auto-distributed: boundary, exclusions, target_particles.

        Attributes
        ----------
        data : dict
            Data object containing all settings and results.
            See the data structure :ref:`here <field_data_structure>`.
        solved : bool
            Boolean to indicate if the field has been solved.

        """
        self._initialised = False
        _auto_distribute = True

        # Check inputs.
        types = [
            gp.sequence.Sequence,
            gp.sequence.SequenceResults,
            gp.mesh.Mesh,
            gp.mesh.MeshResults,
        ]
        self._report(gp.check._check_type(series, "series", types), "TypeError")
        check = gp.check._check_type(target_particles, "target_particles", [int])
        if check:
            try:
                target_particles = int(target_particles)
                self._report(
                    gp.check._conversion(target_particles, "target_particles", int),
                    "Warning",
                )
            except Exception:
                self._report(check, "TypeError")
        self._report(
            gp.check._check_range(target_particles, "target_particles", 1), "ValueError"
        )
        self._report(
            gp.check._check_type(
                boundary,
                "boundary",
                [
                    gp.geometry.region.Circle,
                    gp.geometry.region.Path,
                    np.ndarray,
                    type(None),
                ],
            ),
            "TypeError",
        )
        if type(boundary) == np.ndarray:
            self._report(gp.check._check_dim(boundary, "boundary", 2), "ValueError")
            self._report(
                gp.check._check_axis(boundary, "boundary", 1, [2]), "ValueError"
            )
        check = gp.check._check_type(exclusions, "exclusions", [list])
        if check:
            try:
                exclusions = list(exclusions)
                self._report(
                    gp.check._conversion(exclusions, "exclusions", list, False),
                    "Warning",
                )
            except Exception:
                self._report(check, "TypeError")
        for exclusion in exclusions:
            self._report(
                gp.check._check_type(
                    exclusion,
                    "exclusion",
                    [
                        gp.geometry.region.Circle,
                        gp.geometry.region.Path,
                        np.ndarray,
                        type(None),
                    ],
                ),
                "TypeError",
            )
            if type(exclusion) == np.ndarray:
                self._report(
                    gp.check._check_dim(exclusion, "exclusion", 2), "ValueError"
                )
                self._report(
                    gp.check._check_axis(exclusion, "exclusion", 1, [2]), "ValueError"
                )
        check = gp.check._check_type(
            coordinates, "coordinates", [np.ndarray, type(None)]
        )
        if check:
            try:
                coordinates = np.asarray(coordinates)
                self._report(
                    gp.check._conversion(coordinates, "coordinates", np.ndarray, False),
                    "Warning",
                )
            except Exception:
                self._report(check, "TypeError")
        if coordinates is not None:
            self._report(
                gp.check._check_dim(coordinates, "coordinates", 2), "ValueError"
            )
            self._report(
                gp.check._check_axis(coordinates, "coordinates", 1, [2]), "ValueError"
            )
        check = gp.check._check_type(volumes, "volumes", [np.ndarray, type(None)])
        if check:
            try:
                volumes = np.asarray(volumes)
                self._report(
                    gp.check._conversion(volumes, "volumes", np.ndarray, False),
                    "Warning",
                )
            except Exception:
                self._report(check, "TypeError")
        if volumes:
            self._report(gp.check._check_dim(volumes, "volumes", 1), "ValueError")
            self._report(
                gp.check._check_axis(volumes, "volumes", 0, [np.shape(coordinates)[0]]),
                "ValueError",
            )
        check = gp.check._check_type(stresses, "stresses", [np.ndarray])
        if check:
            try:
                stresses = np.asarray(stresses)
                self._report(
                    gp.check._conversion(stresses, "stresses", np.ndarray, False),
                    "Warning",
                )
            except Exception:
                self._report(check, "TypeError")
        if coordinates is not None:
            self._report(
                gp.check._check_axis(
                    stresses, "stresses", 0, [6, 2, np.shape(coordinates)[0]]
                ),
                "ValueError",
            )
        else:
            self._report(
                gp.check._check_axis(stresses, "stresses", 0, [6, 2]), "ValueError"
            )
        self._report(gp.check._check_type(track, "track", [bool]), "TypeError")
        check = gp.check._check_type(ID, "ID", [str])
        if check:
            try:
                ID = str(ID)
            except Exception:
                self._report(check, "TypeError")

        # Store.
        self._series = series
        self._update_cm(0)
        self._image_0 = self._cm.data["images"]["f_img"]
        self._track = track
        self._strains = strains
        self.solved = False
        self._unsolvable = False
        self._ID = ID
        self._calibrated = series.data["calibrated"]
        if self._calibrated is True:
            self._nref = "Nodes"
        else:
            self._nref = "nodes"
        self._reference_update_register = []
        if self._series.data["type"] == "Sequence":
            self._number_images = len(self._series.data["meshes"]) + 1
        else:
            self._number_images = 2

        if coordinates is not None:
            _auto_distribute = False
            # boundary_hull = sp.spatial.Delaunay(
            #     self._cm.data[self._nref][self._cm.data["boundary"]]
            # )
            # exclusion_hulls = []
            # for exclusion in self._cm.data[self._nref][self._cm.data["exclusions"]]:
            #     exclusion_hulls.append(sp.spatial.Delaunay(exclusion))
            # for coord in coordinates:
            #     if boundary_hull.find_simplex(coord) < 0:
            #         log.error(
            #             (
            #                 "User-specified coordinate {} "
            #                 "outside mesh boundary:\n{}"
            #             ).format(
            #                 coord,
            #                 self._cm.data[self._nref][self._cm.data["boundary"]]
            #             )
            #         )
            #         raise ValueError(
            #             (
            #                 "User-specified coordinate {} outside mesh boundary:\n{}"
            #             ).format(
            #                 coord,
            #                 self._cm.data[self._nref][self._cm.data["boundary"]]
            #             )
            #         )
            #     for i in range(len(exclusion_hulls)):
            #         if exclusion_hulls[i].find_simplex(coord) >= 0:
            #             centre = np.round(
            #                 gp.geometry.utilities.polycentroid(
            #                     self._cm.data[self._nref][
            #                         self._cm.data["exclusions"][i]
            #                     ]
            #                 ),
            #                 2,
            #             )
            #             log.error(
            #                 (
            #                     "User-specified coordinate {} "
            #                     "inside mesh exclusion centred at {}."
            #                 ).format(coord, centre)
            #             )
            #             raise ValueError(
            #                 (
            #                     "User-specified coordinate {} "
            #                     "inside mesh exclusion centred at {}."
            #                 ).format(coord, centre)
            #             )
            if volumes is None:
                volumes = np.ones(np.shape(coordinates)[0])
            self._target_particles = np.shape(coordinates)[0]
        else:
            self._target_particles = target_particles

        # Particle distribution.
        if _auto_distribute is True:
            if boundary is not None:
                if type(boundary) == np.ndarray:
                    self._boundary = boundary
                else:
                    try:
                        self._boundary = boundary.data[self._nref][0]
                    except Exception:
                        log.error(
                            (
                                "Calibration mismatch. Series and boundary object do "
                                "not share the same calibration status."
                            )
                        )
            else:
                self._boundary = self._cm.data[self._nref][self._cm.data["boundary"]]
            if bool(exclusions) is True:
                if type(exclusions[0]) == np.ndarray:
                    self._exclusions = exclusions
                else:
                    try:
                        self._exclusions = []
                        for i in range(len(exclusions)):
                            self._exclusions.append(exclusions[i].data[self._nref][0])
                    except Exception:
                        print(traceback.format_exc())
                        log.error(
                            (
                                "Calibration mismatch. Series and exclusion object do "
                                "not share the same calibration status."
                            )
                        )
            else:
                self._exclusions = self._cm.data[self._nref][
                    self._cm.data["exclusions"]
                ]

            self._size_upper_bound = np.max(
                np.sqrt(
                    np.sum(
                        np.square(
                            np.diff(
                                self._cm.data[self._nref][self._cm.data["boundary"]],
                                axis=0,
                            )
                        ),
                        axis=1,
                    )
                )
            )
            self._size_lower_bound = self._size_upper_bound / 1000
            (
                self._borders,
                self._segments,
                self._curves,
            ) = gp.geometry.meshing._define_RoI(
                boundary=self._boundary,
                exclusions=self._exclusions,
            )
            # Initialize gmsh if not already initialized.
            if gmsh.isInitialized() == 0:
                gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", 2)
            log.info(
                "Generating mesh using gmsh with approximately {n} particles.".format(
                    n=self._target_particles
                )
            )
            self._initial_mesh()
            self._update_mesh()
            gmsh.finalize()
            self._distribute_particles()
            self._stress_state(stresses)
            log.info(
                "Field generated with {p} particles.".format(p=len(self._coordinates))
            )
            self._field = {
                "nodes": self._nodes,
                "elements": self._elements,
                "coordinates": self._coordinates,
                "volumes": self._volumes,
                "stresses": self._stresses,
                "strains": self._strains,
            }
        else:
            self._coordinates = coordinates
            self._volumes = volumes
            self._stress_state(stresses)
            log.info("Using user-specified field.")
            self._field = {
                "coordinates": self._coordinates,
                "volumes": self._volumes,
                "stresses": self._stresses,
            }

        self.data = {
            "type": "Field",
            "ID": self._ID,
            "solved": self.solved,
            "calibrated": self._calibrated,
            "number_images": self._number_images,
            "track": self._track,
            "target_particles": self._target_particles,
            "image_0": self._image_0,
            "field": self._field,
        }

        self._initialised = True

    def _initial_mesh(self):
        """

        Private method to optimize the element size to generate
        approximately the desired number of elements.

        """

        def f(size):
            return self._uniform_remesh(
                size,
                self._borders,
                self._segments,
                self._curves,
                self._target_particles,
                self._size_lower_bound,
                self._size_upper_bound,
            )

        minimize_scalar(
            f,
            bounds=(self._size_lower_bound, self._size_upper_bound),
            method="bounded",
        )

    def _update_mesh(self):
        """

        Private method to update the mesh variables.

        """
        (
            _,
            nc,
            _,
        ) = gmsh.model.mesh.getNodes()  # Extracts: node coordinates.
        _, _, ent = gmsh.model.mesh.getElements(dim=2)  # Extracts: element node tags.
        self._nodes = np.column_stack(
            (nc[0::3], nc[1::3])
        )  # Nodal coordinate array (x,y).
        self._elements = np.reshape(
            (np.asarray(ent) - 1).flatten(), (-1, 3)
        )  # Element connectivity array.

    def _distribute_particles(self):
        self._coordinates = np.mean(self._nodes[self._elements[:, :3]], axis=1)
        M = np.ones((len(self._elements[:, :3]), 3, 3))
        M[:, 1] = self._nodes[self._elements[:, :3]][:, :, 0]
        M[:, 2] = self._nodes[self._elements[:, :3]][:, :, 1]
        self._volumes = abs(0.5 * np.linalg.det(M))

    def _update_cm(self, index):
        try:
            del self._cm
            ("Success")
        except Exception:
            pass
        if self._series.data["type"] == "Sequence":
            if self._series.data["file_settings"]["save_by_reference"] is True:
                self._cm = self._series._load_mesh(index, obj=True, verbose=False)
            else:
                self._cm = self._series.data["meshes"][index]
        else:
            self._cm = self._series

    def _create_rm(self):
        if self._series.data["type"] == "Sequence":
            if self._series.data["file_settings"]["save_by_reference"] is True:
                self._rm = self._series._load_mesh(0, obj=True, verbose=False)
            else:
                self._rm = self._series.data["meshes"][0]
        else:
            self._rm = self._series

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

    def solve(self, *, model=None, state=None, parameters=None, factor=0):
        """
        Method to solve for the field.

        Returns
        -------
        solved : bool
            Boolean to indicate if the particle instances have been solved.
        """
        particle_no = np.shape(self._coordinates)[0]
        self._particles = np.empty(particle_no, dtype=dict)
        self._vol_totals = np.zeros(self.data["number_images"])
        self._create_rm()
        with alive_bar(
            particle_no,
            dual_line=True,
            bar="blocks",
            title="Instantiating particles ...",
        ) as bar:
            for i in range(particle_no):
                self._particles[i] = gp.particle.Particle(
                    series=self._series,
                    coordinate=self._coordinates[i],
                    volume=self._volumes[i],
                    stress=self._stresses[i],
                    warp=self._strains,
                    track=self._track,
                    field=True,
                    ID=str(i),
                )
                bar()
        with alive_bar(
            self.data["number_images"] - 1,
            dual_line=True,
            bar="blocks",
            title="Solving particles for mesh...",
        ) as bar:
            for i in range(self.data["number_images"] - 1):
                self._update_cm(i)
                self._check_update(i)
                for j in range(particle_no):
                    self.solved += self._particles[j]._strain_path_inc(
                        i,
                        self._cm,
                        self._rm,
                    )
                    if i == self.data["number_images"] - 2:
                        self.solved += self._particles[j].solve(
                            model=model,
                            state=state,
                            parameters=parameters,
                            factor=factor,
                        )
                if bool(self.solved) is False:
                    self._unsolvable = True
                    self.data["unsolvable"] = self._unsolvable
                    return self.solved
                bar()

        for i in range(particle_no):
            self._vol_totals += self._particles[i].data["results"]["volumes"]

        if model is not None:
            self._works = np.zeros(self.data["number_images"])
            for i in range(particle_no):
                self._works += self._particles[i].data["results"]["works"]
            self.data.update({"works": self._works})

        self.data.update(
            {
                "particles": self._particles,
                "volumes": self._vol_totals,
                "reference_update_register": self._reference_update_register,
            }
        )
        self.solved = True
        self.data["solved"] = self.solved
        return self.solved

    def stress(self, model=None, state=None, parameters=None):
        self.solved = False
        self.data["solved"] = self.solved
        particle_no = np.shape(self._coordinates)[0]
        for j in range(particle_no):
            self.solved += self._particles[j].solve(
                model=model,
                state=state,
                parameters=parameters,
            )
        if bool(self.solved) is False:
            self._unsolvable = True
            self.data["unsolvable"] = self._unsolvable
            return self.solved
        self._works = np.zeros(self.data["number_images"])
        for i in range(particle_no):
            self._works += self._particles[i].data["results"]["works"]
        self.data.update(
            {
                "works": self._works,
                "particles": self._particles.data,
            }
        )
        self.solved = True
        self.data["solved"] = self.solved
        return self.solved

    def _stress_state(self, stresses):
        """
        Private method to define the initial stress state.
        """
        if np.shape(stresses)[0] == np.shape(self._coordinates)[0]:
            self._stresses = stresses
        elif np.shape(stresses)[0] == 2:
            self._stresses = stresses[0] + (stresses[1] - stresses[0]) * (
                (
                    (self._coordinates[:, 1] - np.min(self._boundary._boundary[:, 1]))
                    / (
                        np.max(self._boundary._boundary[:, 1])
                        - np.min(self._boundary._boundary[:, 1])
                    )
                )[:, np.newaxis]
            )
        else:
            self._stresses = np.tile(stresses, (np.shape(self._coordinates)[0], 1))

    @staticmethod
    def _uniform_remesh(
        size,
        boundary,
        segments,
        curves,
        target_particles,
        size_lower_bound,
        size_upper_bound,
    ):
        """

        Private method to create the initial mesh.

        Parameters
        ----------
        size : int
            Target size of elements.
        boundary : `numpy.ndarray` (Nx,Ny)
            Array of coordinates to define the mesh boundary.
        segments : `numpy.ndarray` (Nx,Ny)
            Array of segments for gmsh mesh generation.
        curves : `numpy.ndarray` (Nx,Ny)
            Array of curves for gmsh mesh generation.
        target_particles : int
            Target number of nodes.
        size_lower_bound : int
            Lower bound on element size.


        Returns
        -------
        error : int
            Error between target and actual number of nodes.

        """
        # Make mesh.
        gmsh.model.add("base")  # Create model.

        # Add points.
        for i in range(np.shape(boundary)[0]):
            gmsh.model.occ.addPoint(boundary[i, 0], boundary[i, 1], 0, size, i)

        # Add line segments.
        for i in range(np.shape(segments)[0]):
            gmsh.model.occ.addLine(segments[i, 0], segments[i, 1], i)

        # Add curves.
        for i in range(len(curves)):
            gmsh.model.occ.addCurveLoop(curves[i], i)
        curve_indices = list(np.arange(len(curves), dtype=np.int32))

        # Create surface.
        gmsh.model.occ.addPlaneSurface(curve_indices, 0)

        # Generate mesh.
        gmsh.option.setNumber("Mesh.MeshSizeMin", size_lower_bound)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size_upper_bound)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize()
        gmsh.model.mesh.setOrder(1)

        # Get mesh topology.
        _, _, ent = gmsh.model.mesh.getElements(dim=2)
        elements = np.reshape((np.asarray(ent) - 1).flatten(), (-1, 3))
        error = (np.shape(elements)[0] - target_particles) ** 2
        return error


class FieldResults(FieldBase):
    """

    FieldResults class for geopyv.

    Parameters
    ----------
    data : dict
        geopyv data dict from Field object.


    Attributes
    ----------
    data : dict
        geopyv data dict from Field object.

    """

    def __init__(self, data):
        """Initialisation of geopyv FieldResults class."""
        self.data = data
