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
            log.info("Tracing particle...")
            obj_type = "Particle"
            data = self.data["particles"][particle_index]
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

        check = gp.check._check_type(boundary, "boundary", [np.ndarray, type(None)])
        if check:
            try:
                boundary = np.asarray(boundary)
                self._report(
                    gp.check._conversion(boundary, "boundary", np.ndarray, False),
                    "Warning",
                )
            except Exception:
                self._report(check, "TypeError")
        if boundary:
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
            check = gp.check._check_type(exclusion, "exclusion", [np.ndarray])
            if check:
                try:
                    exclusion = np.asarray(exclusion)
                    self._report(
                        gp.check._conversion(exclusion, "exclusion", np.ndarray, False),
                        "Warning",
                    )
                except Exception:
                    self._report(check, "TypeError")
            self._report(gp.check._check_dim(exclusion, "exclusion", 2), "ValueError")
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
        if coordinates:
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
        self._report(gp.check._check_type(track, "track", [bool]), "TypeError")

        # Store.
        if series.data["type"] == "Sequence":
            self._series_type = "Sequence"
            if "file_settings" in series.data:
                if series.data["file_settings"]["save_by_reference"]:
                    mesh_no = np.shape(series.data["meshes"])[0]
                    _meshes = np.empty(mesh_no, dtype=object)
                    with alive_bar(
                        mesh_no, dual_line=True, bar="blocks", title="Loading meshes..."
                    ) as bar:
                        for i in range(mesh_no):
                            _meshes[i] = gp.io.load(
                                filename=series.data["file_settings"]["mesh_dir"]
                                + series.data["meshes"][i],
                                verbose=False,
                            ).data
                            bar()
                else:
                    _meshes = series.data["meshes"]
            else:
                _meshes = series.data["meshes"]
            series.data["meshes"] = _meshes
            series.data["file_settings"]["save_by_reference"] = False
            mesh_0 = series.data["meshes"][0]
            self._number_images = np.shape(series.data["meshes"])[0] + 1
        else:
            self._series_type = "Mesh"
            mesh_0 = series.data
            self._number_images = 2
        self._image_0 = mesh_0["images"]["f_img"]
        self._series = series
        self._track = track
        self.solved = False
        self._unsolvable = False

        if coordinates is not None:
            _auto_distribute = False
            image = gp.image.Image(self._image_0)
            for coord in coordinates:
                if (
                    coord[0] > np.shape(image.image_gs)[0]
                    or coord[0] < 0
                    or coord[1] > np.shape(image.image_gs)[1]
                    or coord[1] < 0
                ):
                    log.error(
                        (
                            "User-specified coordinate {value} "
                            "outside image boundary."
                        ).format(value=coord)
                    )
                    raise ValueError(
                        (
                            "User-specified coordinate {value} "
                            "outside image boundary."
                        ).format(value=coord)
                    )
            del image
            if volumes is None:
                volumes = np.ones(np.shape(coordinates)[0])
            self._target_particles = np.shape(coordinates)[0]
        else:
            self._target_particles = target_particles

        self.data = {
            "type": "Field",
            "solved": self.solved,
            "series_type": self._series_type,
            "number_images": self._number_images,
            "track": self._track,
            "target_particles": self._target_particles,
            "image_0": self._image_0,
        }

        # Particle distribution.
        if _auto_distribute is True:
            # Extract region of interest.
            if boundary is None:
                self._boundary = mesh_0["boundary"]
            if exclusions == []:
                self._exclusions = mesh_0["exclusions"]
            self._size_lower_bound = mesh_0["size_lower_bound"]
            self._size_upper_bound = mesh_0["size_upper_bound"]

            self._size_upper_bound = min(
                self._size_upper_bound,
                np.max(
                    np.sqrt(np.sum(np.square(np.diff(self._boundary, axis=0)), axis=1))
                ),
            )

            # Define region of interest.
            (
                self._boundary,
                self._segments,
                self._curves,
                _,
            ) = gp.geometry.meshing._define_RoI(
                gp.image.Image(self._image_0),
                self._boundary,
                self._exclusions,
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
            self._distribute_particles()
            log.info(
                "Field generated with {p} particles.".format(p=len(self._coordinates))
            )
            self._field = {
                "nodes": self._nodes,
                "elements": self._elements,
                "coordinates": self._coordinates,
                "volumes": self._volumes,
            }
        else:
            self._coordinates = coordinates
            self._volumes = volumes
            log.info("Using user-specified field.")
            self._field = {
                "coordinates": self._coordinates,
                "volumes": self._volumes,
            }

        self.data.update({"field": self._field})
        self._initialised = True

    def _initial_mesh(self):
        """

        Private method to optimize the element size to generate
        approximately the desired number of elements.

        """

        def f(size):
            return self._uniform_remesh(
                size,
                self._boundary,
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
        self._update_mesh()
        gmsh.finalize()

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

    def solve(self):
        """
        Method to solve for the field.

        Returns
        -------
        solved : bool
            Boolean to indicate if the particle instances have been solved.
        """
        particle_no = np.shape(self._coordinates)[0]
        self._particles = np.empty(particle_no, dtype=dict)
        with alive_bar(
            particle_no, dual_line=True, bar="blocks", title="Solving particles..."
        ) as bar:
            for i in range(particle_no):
                particle = gp.particle.Particle(
                    series=self._series,
                    coordinate_0=self._coordinates[i],
                    volume_0=self._volumes[i],
                    track=self._track,
                )
                _particle_solved = particle.solve()
                if _particle_solved is False:
                    self._unsolvable = True
                    self.data["unsolvable"] = self._unsolvable
                    return self.solved
                self._particles[i] = particle.data
                del particle
                bar()
        self.data.update({"particles": self._particles})
        self.solved = True
        self.data["solved"] = self.solved

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
