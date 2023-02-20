"""

Field module for geopyv.

"""
import logging
import numpy as np
import geopyv as gp
from geopyv.object import Object
import gmsh
from scipy.optimize import minimize_scalar
import PIL.Image as ImagePIL
import PIL.ImageDraw as ImageDrawPIL

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
        """
        log.info("Inspecting field...")
        fig, ax = gp.plots.inspect_field(
            self.data, mesh=mesh, show=show, block=block, save=save
        )
        return fig, ax

    def volume_divergence(self, show=True, block=True, save=None):
        """
        Method to show the volumetric error in the particle field.
        """

    def trace(
        self,
        quantity="warps",
        component=0,
        start_frame=None,
        end_frame=None,
        imshow=True,
        colorbar=True,
        ticks=None,
        alpha=0.75,
        axis=None,
        xlim=None,
        ylim=None,
        show=True,
        block=True,
        save=None,
    ):
        if quantity is not None:
            log.info("Tracing field...")
            fig, ax = gp.plots.trace_particle(
                data=self.data,
                quantity=quantity,
                component=component,
                start_frame=start_frame,
                end_frame=end_frame,
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


class Field(FieldBase):
    def __init__(
        self,
        *,
        series=None,
        target_particles=1000,
        moving=True,
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
        target_nodes : int, optional
            Target number of particles. Defaults to a value of 1000.
        moving : bool, optional
            Boolean to specify if particles should move or remain static. True - move (Lagrangian), False - static (Eularian). Defaults to True.
        boundary : numpy.ndarray (N,2), optional
            Array of coordinates to define the particle auto-distribution mesh boundary.
        exclusions : list, optional
            List of `numpy.ndarray` (N,2) to define the particle auto-distribution mesh exclusions.
        coordinates : numpy.ndarray (N,2), optional
            Array of coordinates to define the initial particle positions.
        volumes : numpy.ndarray (N,), optional
            Array of volumes for particle representation. Defaults to np.ones(N) i.e. measure of volumetric strain.

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
            See the data structure :ref:`here <mesh_data_structure>`.
        solved : bool
            Boolean to indicate if the mesh has been solved.

        """
        self._initialised = False
        _auto_distribute = True

        # Check types
        if series.data["type"] != "Sequence" and series.data["type"] != "Mesh":
            log.error(
                "Invalid series type. Must be gp.sequence.Sequence or gp.mesh.Mesh."
            )
        if type(moving) != bool:
            log.error("Invalid moving type. Must be a bool.")
        if series.data["type"] == "Sequence":
            self._series_type = "Sequence"
            mesh_0 = series.data["meshes"][0]
            self._number_images = self._series.data["number_images"]
        else:
            self._series_type = "Mesh"
            mesh_0 = series.data
            self._number_images = 1
        self._image_0 = mesh_0["images"]["f_img"]
        self._moving = moving
        self._series = series
        self.solved = False
        self._unsolvable = False

        if coordinates is not None:
            _auto_distribute = False
            if type(coordinates) != np.ndarray:
                try:
                    coordinates = np.asarray(coordinates)
                except:
                    log.error(
                        "Coordinates array type invalid. Expected a numpy.ndarray."
                    )
                    return False
            if np.shape(coordinates)[1] != 2:
                log.error(
                    "Coordinates array shape invalid. Expected a (N,2) numpy.ndarray."
                )
                return False
            if np.ndim(coordinates) != 2:
                log.error(
                    "Coordinates array dimensions invalid. Expected a 2D numpy.ndarray."
                )
            image = gp.image.Image(self._image_0)
            for coord in coordinates:
                if (
                    coord[0] > np.shape(image.image_gs)[0]
                    or coord[0] < 0
                    or coord[1] > np.shape(image.image_gs)[1]
                    or coord[1] < 0
                ):
                    log.error("User-specified coordinate outside image boundary.")
                    return False
            del image
            if volumes is not None:
                if type(volumes) != np.ndarray:
                    try:
                        volumes = np.asarray(volumes)
                    except:
                        log.error(
                            "Volumes array type invalid. Expected a numpy.ndarray."
                        )
                        return False
                if np.ndim(volumes) != 1:
                    log.error(
                        "Volumes array dimensions invalid. Expected a 1D numpy.ndarray."
                    )
                    return False
                if np.shape(volumes)[0] != np.shape(coordinates)[0]:
                    log.error(
                        "Volumes-coordinates array mismatch. {volumes} volumes given for {coordinates} coordinates.".format(
                            volumes=np.shape(volumes)[0],
                            coordinates=np.shape(coordinates)[0],
                        )
                    )
                    return False
            else:
                volumes = np.ones(np.shape(coordinates)[0])
            self._target_particles = np.shape(coordinates)[0]
        else:
            if type(target_particles) != int:
                try:
                    volumes = int(target_particles)
                except:
                    log.error("Target particles type invalid. Expected an integer.")
                    return False
            if target_particles < 0:
                log.error("Target particles out of range. Must be >0.")
            if boundary is not None:
                if type(boundary) != np.ndarray:
                    try:
                        boundary = np.asarray(boundary)
                    except:
                        log.error(
                            "Boundary array type invalid. Expected a numpy.ndarray."
                        )
                        return False
                if np.shape(boundary)[1] != 2:
                    log.error(
                        "Boundary array shape invalid. Expected a (N,2) numpy.ndarray."
                    )
                    return False
                if np.ndim(boundary) != 2:
                    log.error(
                        "Boundary array dimensions invalid. Expected a 2D numpy.ndarray."
                    )
            if type(exclusions) != list:
                log.error("Exclusions type invalid. Expected a list.")
                return False
            for exclusion in exclusions:
                if type(exclusion) != np.ndarray:
                    log.error(
                        "Exclusion coordinate array type invalid. Expected a numpy.ndarray."
                    )
                    return False
                if np.ndim(exclusion) != 2:
                    log.error(
                        "Exclusion array dimensions invalid. Expected a 2D numpy.ndarray."
                    )
                    return False
                if np.shape(exclusion)[1] != 2:
                    log.error(
                        "Exclusion coordinate array shape invalid. Must be numpy.ndarray of size (n, 2)."
                    )
                    return False

            self._target_particles = target_particles

        self.data = {
            "type": "Field",
            "solved": self.solved,
            "series_type": self._series_type,
            "number_images": self._number_images,
            "moving": self._moving,
            "target_particles": self._target_particles,
            "image_0": self._image_0,
        }

        # Particle distribution.
        if _auto_distribute == True:
            # Extract region of interest.
            if boundary is None:
                self._boundary = mesh_0["boundary"]
            if exclusions == []:
                self._exclusions = mesh_0["exclusions"]
            self._size_lower_bound = mesh_0["size_lower_bound"]
            self._size_upper_bound = mesh_0["size_upper_bound"]

            # Define region of interest.

            (
                self._boundary,
                self._segments,
                self._curves,
                _,
            ) = gp.geometry.meshing._define_RoI(
                gp.image.Image(self._image_0), self._boundary, self._exclusions
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
            self._field = {"coordinates": self._coordinates, "volumes": self._volumes}

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
            f, bounds=(self._size_lower_bound, self._size_upper_bound), method="bounded"
        )
        self._update_mesh()
        gmsh.finalize()

    def _update_mesh(self):
        """

        Private method to update the mesh variables.

        """
        _, nc, _ = gmsh.model.mesh.getNodes()  # Extracts: node coordinates.
        _, _, ent = gmsh.model.mesh.getElements(dim=2)  # Extracts: element node tags.
        self._nodes = np.column_stack(
            (nc[0::3], nc[1::3])
        )  # Nodal coordinate array (x,y).
        self._elements = np.reshape(
            (np.asarray(ent) - 1).flatten(), (-1, 6)
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

        self._particles = np.empty(len(self._coordinates), dtype=dict)
        for i in range(len(self._coordinates)):
            particle = gp.particle.Particle(
                series=self._series,
                coordinate_0=self._coordinates[i],
                volume_0=self._volumes[i],
                moving=self._moving,
            )
            _particle_solved = particle.solve()
            if _particle_solved == False:
                self._unsolvable = True
                self.data["unsolvable"] = self._unsolvable
                return self.solved
            self._particles[i] = particle.data["results"]
            del particle
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
        target_nodes : int
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
        gmsh.model.mesh.setOrder(2)

        # Get mesh topology.
        _, _, ent = gmsh.model.mesh.getElements(dim=2)
        elements = np.reshape((np.asarray(ent) - 1).flatten(), (-1, 6))
        number_particles = len(elements)
        error = abs(number_particles - target_particles)
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
