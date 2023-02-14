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
    ):
        self._initialised = False
        # Check types
        if series.data["type"] != "Sequence" and series.data["type"] != "Mesh":
            log.error(
                "Invalid series type. Must be gp.sequence.Sequence or gp.mesh.Mesh."
            )
        if type(target_particles) != int:
            log.error("Maximum number of nodes not of integer type.")
        elif target_particles < 0:
            log.error("Target number of particles must be more than 0.")
        if type(moving) != bool:
            log.error("Invalid moving type. Must be a bool.")

        self._target_particles = target_particles
        self._moving = moving
        self.solved = False
        self._series = series
        if series.data["type"] == "Sequence":
            self._series_type = "Sequence"
            mesh_0 = series.data["meshes"][0]
            self._number_images = self._series.data["number_images"]
        else:
            self._series_type = "Mesh"
            mesh_0 = series.data
            self._number_images = 1
        self._image_0 = mesh_0["images"]["f_img"]

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

        self.data = {
            "type": "Field",
            "solved": self.solved,
            "series_type": self._series_type,
            "number_images": self._number_images,
            "moving": self._moving,
            "target_particles": self._target_particles,
            "image_0": self._image_0,
        }
        self._initial_mesh()
        self._distribute_particles()
        log.info("Field generated with {p} particles.".format(p=len(self._coordinates)))
        self._mesh = {
            "nodes": self._nodes,
            "elements": self._elements,
            "coordinates": self._coordinates,
        }
        self.data.update({"mesh": self._mesh})
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
        self._particles = np.empty(len(self._coordinates), dtype=dict)
        for i in range(len(self._coordinates)):
            particle = gp.particle.Particle(
                series=self._series,
                coordinate_0=self._coordinates[i],
                volume_0=self._volumes[i],
                moving=self._moving,
            )
            particle.solve()
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
