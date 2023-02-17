"""

Mesh module for geopyv.

"""
import logging
import numpy as np
import scipy as sp
import geopyv as gp
from geopyv.object import Object
import gmsh
from copy import deepcopy
from scipy.optimize import minimize_scalar
import PIL.Image as ImagePIL
import PIL.ImageDraw as ImageDrawPIL
from alive_progress import alive_bar

log = logging.getLogger(__name__)


class MeshBase(Object):
    """

    Mesh base class to be used as a mixin. Contains plot functionality.

    """

    def __init__(self):
        super().__init__(object_type="Mesh")
        """

        Mesh base class initialiser.

        """

    def inspect(self, subset=None, show=True, block=True, save=None):
        """

        Method to show the mesh and associated subset quality metrics.

        """
        # If a subset index is given, inspect the subset.
        if subset is not None:
            if subset >= 0 and subset < len(self.data["results"]["subsets"]):
                subset_data = self.data["results"]["subsets"][subset]
                mask = self.data["mask"]
                fig, ax = gp.plots.inspect_subset(
                    data=subset_data, mask=mask, show=show, block=block, save=save
                )
                return fig, ax
            else:
                log.error(
                    "Subset index is out of the range of the mesh object contents."
                )
        # Otherwise inspect the mesh.
        else:
            fig, ax = gp.plots.inspect_mesh(
                data=self.data, show=show, block=block, save=save
            )
            return fig, ax

    def convergence(self, subset=None, quantity=None, show=True, block=True, save=None):
        """

        Method to plot the rate of convergence for the mesh or subset.

        """
        # If a subset index is given, inspect the subset.
        if subset is not None:
            if subset >= 0 and subset < len(self.data["results"]["subsets"]):
                fig, ax = gp.plots.convergence_subset(
                    self.data["results"]["subsets"][subset],
                    show=show,
                    block=block,
                    save=save,
                )
                return fig, ax
            else:
                log.error(
                    "Subset index is out of the range of the mesh object contents."
                )
        # Otherwise inspect the mesh.
        else:
            if quantity is None:
                quantity = "C_ZNCC"
            fig, ax = gp.plots.convergence_mesh(
                data=self.data, quantity=quantity, show=show, block=block, save=save
            )
            return fig, ax

    def contour(
        self,
        quantity="C_ZNCC",
        imshow=True,
        colorbar=True,
        ticks=None,
        mesh=False,
        alpha=0.75,
        levels=None,
        axis=None,
        xlim=None,
        ylim=None,
        show=True,
        block=True,
        save=None,
    ):
        """

        Method to plot the contours of a given measure.

        """
        if quantity is not None:
            fig, ax = gp.plots.contour_mesh(
                data=self.data,
                imshow=imshow,
                quantity=quantity,
                colorbar=colorbar,
                ticks=ticks,
                mesh=mesh,
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

    def quiver(
        self,
        scale=1,
        imshow=True,
        mesh=False,
        axis=None,
        xlim=None,
        ylim=None,
        show=True,
        block=True,
        save=None,
    ):
        """

        Method to plot a quiver plot of the displacements.

        """
        fig, ax = gp.plots.quiver_mesh(
            data=self.data,
            scale=scale,
            imshow=imshow,
            mesh=mesh,
            axis=axis,
            xlim=xlim,
            ylim=ylim,
            show=show,
            block=block,
            save=save,
        )
        return fig, ax


class Mesh(MeshBase):
    def __init__(
        self,
        *,
        f_img=None,
        g_img=None,
        target_nodes=1000,
        boundary=None,
        exclusions=[],
        size_lower_bound=1,
        size_upper_bound=1000,
        mesh_order=2,
    ):
        """

        Mesh class for geopyv.

        Parameters
        ----------
        f_img : geopyv.image.Image, optional
            Reference image of geopyv.image.Image class, instantiated
            by :mod:`~geopyv.image.Image`.
        g_img : geopyv.image.Image, optional
            Target image of geopyv.imageImage class, instantiated by
            :mod:`~geopyv.image.Image`.
        target_nodes : int, optional
            Target number of nodes.
        boundary : `numpy.ndarray` (Nx,Ny)
            Array of coordinates to define the mesh boundary.
        exclusions : list, optional
            List of `numpy.ndarray` to define the mesh exclusions.
        size_lower_bound : int, optional
            Lower bound on element size. Defaults to a value of 1.
        upper_lower_bound : int, optional
            Lower bound on element size. Defaults to a value of 1000.


        Attributes
        ----------
        data : dict
            Data object containing all settings and results.
            See the data structure :ref:`here <mesh_data_structure>`.
        solved : bool
            Boolean to indicate if the mesh has been solved.

        """

        self._initialised = False

        # Store variables.
        self._f_img = f_img
        self._g_img = g_img
        self._target_nodes = target_nodes
        self._boundary = boundary
        self._exclusions = exclusions
        self._size_lower_bound = size_lower_bound
        self._size_upper_bound = size_upper_bound
        self._mesh_order = mesh_order
        self.solved = False
        self._unsolvable = False

        # Check types.
        # if type(f_img) != gp.image.Image:
        #     self._f_img = gp.io._load_f_img()
        # if type(g_img) != gp.image.Image:
        #     self._g_img = gp.io._load_g_img()
        if type(target_nodes) != int:
            log.warn(
                "Target node number not of integer type. Attempting integer conversion..."
            )
            try:
                self._target_nodes = int(target_nodes)
            except:
                log.warn(
                    "Target node number conversion failed. Setting target nodes as 1000."
                )
                self._target_nodes = 1000
        elif target_nodes <= 0:
            log.warn("Invalid target nodes number. Setting target nodes as 1000.")
            self._target_nodes = 1000
        if type(boundary) != np.ndarray:
            # self._boundary
            log.warn("Boundary coordinate array of invalid type. Cannot in")
            raise TypeError(
                "Boundary coordinate array of invalid type. Cannot initialise mesh."
            )
        if type(exclusions) != list:
            raise TypeError(
                "Exclusion coordinate array of invalid type. Cannot initialise mesh."
            )
        for exclusion in exclusions:
            if np.shape(exclusion)[1] != 2:
                raise ValueError(
                    "Exclusion coordinate array of invalid shape. "
                    "Must be `numpy.ndarray` of size (n, 2)."
                )

        # Define region of interest.
        self._define_RoI()

        # Initialize gmsh if not already initialized.
        if gmsh.isInitialized() == 0:
            gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)
        # 0: silent except for fatal errors, 1: +errors, 2: +warnings,
        # 3: +direct, 4: +information, 5: +status, 99: +debug.

        # Create initial mesh.
        log.info(
            "Generating mesh using gmsh with approximately {n} nodes.".format(
                n=self._target_nodes
            )
        )
        self._initial_mesh()
        log.info(
            "Mesh generated with {n} nodes and {e} elements.".format(
                n=len(self._nodes), e=len(self._elements)
            )
        )
        gmsh.finalize()

        # Data.
        self.data = {
            "type": "Mesh",
            "solved": self.solved,
            "unsolvable": self._unsolvable,
            "images": {
                "f_img": self._f_img.filepath,
                "g_img": self._g_img.filepath,
            },
            "target_nodes": self._target_nodes,
            "boundary": self._boundary,
            "exclusions": self._exclusions,
            "size_lower_bound": self._size_lower_bound,
            "size_upper_bound": self._size_upper_bound,
            "nodes": self._nodes,
            "elements": self._elements,
            "mask": self._mask,
        }

    def set_target_nodes(self, target_nodes):
        """

        Method to create a mesh with a target number of nodes.

        Parameters
        ----------
        target_nodes : int
            Target number of nodes.


        .. note::
            * This method can be used to update the number of target nodes.
            * It will generate a new initial mesh with the specified target
              number of nodes.

        """
        self._target_nodes = target_nodes

        # Initialize gmsh if not already initialized.
        if gmsh.isInitialized() == 0:
            gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)

        # Create mesh.
        log.info(
            "Generating mesh using gmsh with approximately {n} nodes.".format(
                n=self._target_nodes
            )
        )
        self._initial_mesh()
        log.info(
            "Mesh generated with {n} nodes and {e} elements.".format(
                n=len(self._nodes), e=len(self._elements)
            )
        )
        gmsh.finalize()

    def solve(
        self,
        *,
        seed_coord=None,
        template=None,
        max_norm=1e-3,
        max_iterations=15,
        subset_order=1,
        tolerance=0.7,
        method="ICGN",
        adaptive_iterations=0,
        alpha=0.5,
    ):
        r"""

        Method to solve for the mesh.

        Parameters
        ----------
        max_norm : float, optional
            Exit criterion for norm of increment in warp function. Defaults to value of
            :math:`1 \cdot 10^{-3}`.
        max_iterations : int, optional
            Exit criterion for number of Gauss-Newton iterations. Defaults to value
            of 50.
        order : int
            Warp function order. Options are 1 and 2.
        tolerance: float, optional
            Correlation coefficient tolerance. Defaults to a value of 0.7.
        method : str
            Solution method. Options are FAGN and ICGN.
            Default is ICGN since it is faster.
        adaptive_iterations : int, optional
            Number of mesh adaptivity iterations to perform. Defaults to a value of 0.
        alpha : float, optional
            Mesh adaptivity control parameter. Defaults to a value of 0.5.


        Returns
        -------
        solved : bool
            Boolean to indicate if the subset instance has been solved.

        """
        # Check inputs.
        if type(seed_coord) != np.ndarray:
            raise TypeError(
                "Coordinate is not of `numpy.ndarray` type. Cannot initiate solver."
            )
        elif type(adaptive_iterations) != int:
            raise TypeError(
                "Number of adaptive iterations of invalid type. "
                "Must be an integer greater than or equal to zero."
            )
        if template is None:
            template = gp.templates.Circle(50)
        elif (
            type(template) != gp.templates.Circle
            and type(template) != gp.templates.Square
        ):
            raise TypeError("Template is not a type defined in geopyv.templates.")

        # Store variables.
        self._seed_coord = seed_coord
        self._template = template
        self._max_iterations = max_iterations
        self._max_norm = max_norm
        self._adaptive_iterations = adaptive_iterations
        self._method = method
        self._subset_order = subset_order
        self._tolerance = tolerance
        self._alpha = alpha
        self._subset_bgf_nodes = None
        self._subset_bgf_values = None
        self._update = False
        self._p_0 = np.zeros(6 * self._subset_order)

        # Initialize gmsh if not already initialized.
        if gmsh.isInitialized() == 0:
            gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)

        # Create initial mesh.
        self._initial_mesh()

        # Solve initial mesh.
        self._message = "Solving initial mesh"
        self._find_seed_node()
        try:
            self._reliability_guided()
            if self._unsolvable:
                log.error(
                    "Specified correlation coefficient tolerance not met. "
                    "Minimum correlation coefficient: "
                    "{min_C:.2f}; tolerance: {tolerance:.2f}.".format(
                        min_C=np.amin(self._C_ZNCC[np.where(self._C_ZNCC > 0.0)]),
                        tolerance=self._tolerance,
                    )
                )
                return self.solved
            # Solve adaptive iterations.
            for iteration in range(1, adaptive_iterations + 1):
                self._message = "Adaptive iteration {}".format(iteration)
                self._adaptive_mesh()
                self._update_mesh()
                self._adaptive_subset()
                self._find_seed_node()
                self._reliability_guided()
                if self._unsolvable:
                    log.error(
                        "Specified correlation coefficient tolerance not met. "
                        "Minimum correlation coefficient: "
                        "{min_C:.2f}; tolerance: {tolerance:.2f}.".format(
                            min_C=np.amin(self._C_ZNCC[np.where(self._C_ZNCC > 0.0)]),
                            tolerance=self._tolerance,
                        )
                    )
                    return self.solved
            log.info(
                "Solved mesh. Minimum correlation coefficient: {min_C:.2f}; "
                "maximum correlation coefficient: {max_C:.2f}.".format(
                    min_C=np.amin(self._C_ZNCC), max_C=np.amax(self._C_ZNCC)
                )
            )

            # Pack data.
            self.solved = True
            self.data["nodes"] = self._nodes
            self.data["elements"] = self._elements
            self.data["areas"] = self._areas
            self.data["solved"] = self.solved
            self.data["unsolvable"] = self._unsolvable

            # Pack settings.
            self._settings = {
                "max_iterations": self._max_iterations,
                "max_norm": self._max_norm,
                "adaptive_iterations": self._adaptive_iterations,
                "method": self._method,
                "mesh_order": self._mesh_order,
                "tolerance": self._tolerance,
            }
            self.data.update({"settings": self._settings})

            # Extract data from subsets.
            subset_data = []
            for subset in self._subsets:
                subset_data.append(subset.data)

            # Pack results.
            self._results = {
                "subsets": subset_data,
                "displacements": self._displacements,
                "du": self._du,
                "d2u": self._d2u,
                "C_ZNCC": self._C_ZNCC,
            }
            self.data.update({"results": self._results})

        except Exception:
            log.info("exception")
            self._update = True
            self.solved = False
            self._unsolvable = True
        gmsh.finalize()
        return self.solved

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

        # If track "add".
        _boundary_tags = []
        for i in gmsh.model.getEntitiesForPhysicalGroup(1, 0):
            _boundary_tags.append(
                np.roll(
                    np.asarray(
                        (gmsh.model.mesh.getNodes(1, i, includeBoundary=True)[0] - 1)
                    )[:-1],
                    1,
                )
            )
        self._boundary_add_tags = np.concatenate(_boundary_tags).ravel()
        self._exclusions_add_tags = []
        for i in range(np.shape(self._exclusions)[0]):
            _exclusion_tags = []
            for j in gmsh.model.getEntitiesForPhysicalGroup(1, i + 1):
                _exclusion_tags.append(
                    np.roll(
                        np.asarray(
                            (
                                gmsh.model.mesh.getNodes(1, j, includeBoundary=True)[0]
                                - 1
                            )
                        )[:-1],
                        1,
                    )
                )
            self._exclusions_add_tags.append(np.concatenate(_exclusion_tags).ravel())

        # If track "move".
        self._boundary_move_tags = gmsh.model.occ.getCurveLoops(0)[1][0]
        self._exclusions_move_tags = []
        for i in range(len(self._exclusions)):
            self._exclusions_move_tags.append(gmsh.model.occ.getCurveLoops(0)[1][i + 1])

    def _find_seed_node(self):
        """

        Private method to find seed node given seed coordinate.

        """
        dist = np.sqrt(
            (self._nodes[:, 0] - self._seed_coord[0]) ** 2
            + (self._nodes[:, 1] - self._seed_coord[1]) ** 2
        )
        self._seed_node = np.argmin(dist)

    def _define_RoI(self):
        """

        Private method to define the RoI.

        """
        # Create binary mask RoI.
        binary_img = ImagePIL.new(
            "L",
            (np.shape(self._f_img.image_gs)[1], np.shape(self._f_img.image_gs)[0]),
            0,
        )
        ImageDrawPIL.Draw(binary_img).polygon(
            self._boundary.flatten().tolist(), outline=1, fill=1
        )

        # Create objects for mesh generation.
        self._segments = np.empty(
            (np.shape(self._boundary)[0], 2), dtype=np.int32
        )  # Initiate segment array.
        self._segments[:, 0] = np.arange(
            np.shape(self._boundary)[0], dtype=np.int32
        )  # Fill segment array.
        self._segments[:, 1] = np.roll(self._segments[:, 0], -1)  # Fill segment array.
        self._curves = [list(self._segments[:, 0])]  # Create curve list.

        # Add exclusions.
        self._borders = self._boundary
        for exclusion in self._exclusions:
            ImageDrawPIL.Draw(binary_img).polygon(
                exclusion.flatten().tolist(), outline=1, fill=0
            )  # Add exclusion to binary mask.
            cur_max_idx = np.amax(
                self._segments
            )  # Highest index used by current segments.
            exclusion_segment = np.empty(
                np.shape(exclusion)
            )  # Initiate exclusion segment array.
            exclusion_segment[:, 0] = np.arange(
                cur_max_idx + 1, cur_max_idx + 1 + np.shape(exclusion)[0]
            )  # Fill exclusion segment array.
            exclusion_segment[:, 1] = np.roll(
                exclusion_segment[:, 0], -1
            )  # Fill exclusion segment array.
            self._borders = np.append(
                self._borders, exclusion, axis=0
            )  # Append exclusion to boundary array.
            self._segments = np.append(
                self._segments, exclusion_segment, axis=0
            ).astype(
                "int32"
            )  # Append exclusion segments to segment array.
            self._curves.append(
                list(exclusion_segment[:, 0].astype("int32"))
            )  # Append exclusion curve to curve list.

        # Finalise mask.
        self._mask = np.array(binary_img)

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
                self._target_nodes,
                self._size_lower_bound,
                self._size_upper_bound,
                self._mesh_order,
            )

        minimize_scalar(
            f, bounds=(self._size_lower_bound, self._size_upper_bound), method="bounded"
        )
        self._update_mesh()

    def _adaptive_mesh(self):
        """

        Private method to perform adaptive remeshing.

        """
        message = "Adaptively remeshing..."
        with alive_bar(dual_line=True, bar=None, title=message) as bar:
            D = (
                abs(self._du[:, 0, 1] + self._du[:, 1, 0]) * self._areas
            )  # Elemental shear strain-area products.
            D_b = np.mean(D)  # Mean elemental shear strain-area product.
            self._areas *= (
                np.clip(D / D_b, self._alpha, 1 / self._alpha)
            ) ** -2  # Target element areas calculated.

            def f(scale):
                return self._adaptive_remesh(
                    scale,
                    self._target_nodes,
                    self._nodes,
                    self._elements,
                    self._areas,
                    self._mesh_order,
                )

            minimize_scalar(f)
            bar()

    @staticmethod
    def _uniform_remesh(
        size,
        boundary,
        segments,
        curves,
        target_nodes,
        size_lower_bound,
        size_upper_bound,
        order,
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

        # Add physical groups.
        for i in range(len(curves)):
            gmsh.model.addPhysicalGroup(1, curves[i], i)

        # Mesh.
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize()
        gmsh.model.mesh.setOrder(order)

        # Get mesh topology.
        (
            _,
            nc,
            _,
        ) = (
            gmsh.model.mesh.getNodes()
        )  # Extracts: node tags, node coordinates, parametric coordinates.
        nodes = np.column_stack((nc[0::3], nc[1::3]))  # Nodal coordinate array (x,y).
        number_nodes = len(nodes)
        # error = target_nodes - number_nodes
        error = number_nodes - target_nodes
        return error

    @staticmethod
    def _adaptive_remesh(scale, target, nodes, elements, areas, order):
        """

        Private method to perform adaptive mesh generation.

        Parameters
        ----------
        scale : float
            Scale factor on element size.
        target : int
            Target number of nodes.
        nodes : `numpy.ndarray`
            Nodes for background mesh.
        elements : `numpy.ndarray`
            Elements for background mesh.
        areas : float
            Target element areas.


        Returns
        -------
        error : float
            Error between target and actual number of nodes.

        """
        lengths = gp.geometry.utilities.area_to_length(
            areas * scale
        )  # Convert target areas to target characteristic lengths.
        bg = gmsh.view.add("bg", 1)  # Create background view.
        data = np.pad(
            nodes[elements[:, :3]], ((0, 0), (0, 0), (0, 2)), mode="constant"
        )  # Prepare data input (coordinates and buffer).
        data[:, :, 3] = np.reshape(
            np.repeat(lengths, 3), (-1, 3)
        )  # Fill data input buffer with target weights.
        data = np.transpose(data, (0, 2, 1)).flatten()  # Reshape for input.
        gmsh.view.addListData(bg, "ST", len(elements), data)  # Add data to view.
        bgf = gmsh.model.mesh.field.add("PostView")  # Add view to field.
        gmsh.model.mesh.field.setNumber(
            bgf, "ViewTag", bg
        )  # Establish field reference (important for function reuse).
        gmsh.model.mesh.field.setAsBackgroundMesh(bgf)  # Set field as background.
        gmsh.option.setNumber(
            "Mesh.MeshSizeExtendFromBoundary", 0
        )  # Prevent boundary influence on mesh.
        gmsh.option.setNumber(
            "Mesh.MeshSizeFromPoints", 0
        )  # Prevent point influence on mesh.
        gmsh.option.setNumber(
            "Mesh.MeshSizeFromCurvature", 0
        )  # Prevent curve influence on mesh.
        gmsh.option.setNumber("Mesh.Algorithm", 5)
        gmsh.model.mesh.clear()  # Tidy.
        gmsh.model.mesh.generate(2)  # Generate mesh.
        gmsh.model.mesh.optimize()
        gmsh.model.mesh.setOrder(order)

        (
            nt,
            nc,
            npar,
        ) = (
            gmsh.model.mesh.getNodes()
        )  # Extracts: node tags, node coordinates, parametric coordinates.
        nodes = np.column_stack((nc[0::3], nc[1::3]))  # Nodal coordinate array (x,y).
        error = (np.shape(nodes)[0] - target) ** 2
        return error

    def _adaptive_subset(self):
        """

        Private method to compute adaptive subset size.

        .. warning::
            * Implementation not yet complete.

        """
        subset_bgf = sp.interpolate.RBFInterpolator(
            self._subset_bgf_nodes,
            self._subset_bgf_values,
            neighbors=10,
            kernel="cubic",
        )
        subset_bgf(self._nodes)

    def _update_subset_bgf(self):
        """

        Private method to compute the background mesh.

        """
        if self._subset_bgf_nodes is not None:
            self._subset_bgf_nodes = np.append(
                self._subset_bgf_nodes,
                np.mean(self._nodes[self._elements], axis=1),
                axis=0,
            )
            self._subset_bgf_values = np.append(
                self._subset_bgf_values, np.mean(self._d2u, axis=(1, 2)), axis=0
            )
        else:
            self._subset_bgf_nodes = np.mean(self._nodes[self._elements], axis=1)
            self._subset_bgf_values = np.mean(self._d2u, axis=(1, 2))

    def _element_area(self):
        """
        Private method to calculate the element areas.

        """
        M = np.ones((len(self._elements), 3, 3))
        M[:, 1] = self._nodes[self._elements[:, :3]][
            :, :, 0
        ]  # [:3] will provide corner nodes in both 1st and 2nd order element case.
        M[:, 2] = self._nodes[self._elements[:, :3]][:, :, 1]
        self._areas = 0.5 * np.linalg.det(M)

    def _element_strains(self):
        """

        Private method to calculate the elemental strain the "B" matrix
        relating element node displacements to elemental strain.

        """
        # Local coordinates
        A = np.ones((len(self._elements), 3, 3))
        A[:, :, 1:] = self._nodes[self._elements[:, :3]]

        # Weighting function (and derivatives to 2nd order).
        dN = np.asarray(
            [[1 / 3, 0, -1 / 3, 4 / 3, -4 / 3, 0], [0, 1 / 3, -1 / 3, 4 / 3, 0, -4 / 3]]
        )
        d2N = np.asarray(
            [[4, 0, 4, 0, 0, -8], [0, 0, 4, 4, -4, -4], [0, 4, 4, 0, -8, 0]]
        )

        # 1st Order Strains
        J_x_T = dN @ self._nodes[self._elements]
        J_u_T = dN @ self._displacements[self._elements]
        du = np.linalg.inv(J_x_T) @ J_u_T

        # 2nd Order Strains
        d2udzeta2 = d2N @ self._displacements[self._elements]
        J_zeta = np.zeros((len(self._elements), 2, 2))
        J_zeta[:, 0, 0] = (
            self._nodes[self._elements][:, 1, 1] - self._nodes[self._elements][:, 2, 1]
        )
        J_zeta[:, 0, 1] = (
            self._nodes[self._elements][:, 2, 0] - self._nodes[self._elements][:, 1, 0]
        )
        J_zeta[:, 1, 0] = (
            self._nodes[self._elements][:, 2, 1] - self._nodes[self._elements][:, 0, 1]
        )
        J_zeta[:, 1, 1] = (
            self._nodes[self._elements][:, 0, 0] - self._nodes[self._elements][:, 2, 0]
        )
        J_zeta /= np.linalg.det(A)[:, None, None]
        d2u = np.zeros((len(self._elements), 3, 2))
        d2u[:, 0, 0] = (
            d2udzeta2[:, 0, 0] * J_zeta[:, 0, 0] ** 2
            + 2 * d2udzeta2[:, 1, 0] * J_zeta[:, 0, 0] * J_zeta[:, 1, 0]
            + d2udzeta2[:, 2, 0] * J_zeta[:, 1, 0] ** 2
        )
        d2u[:, 0, 1] = (
            d2udzeta2[:, 0, 1] * J_zeta[:, 0, 0] ** 2
            + 2 * d2udzeta2[:, 1, 1] * J_zeta[:, 0, 0] * J_zeta[:, 1, 0]
            + d2udzeta2[:, 2, 1] * J_zeta[:, 1, 0] ** 2
        )
        d2u[:, 1, 0] = (
            d2udzeta2[:, 0, 0] * J_zeta[:, 0, 0] * J_zeta[:, 0, 1]
            + d2udzeta2[:, 1, 0]
            * (J_zeta[:, 0, 0] * J_zeta[:, 1, 1] + J_zeta[:, 1, 0] * J_zeta[:, 0, 1])
            + d2udzeta2[:, 2, 0] * J_zeta[:, 1, 0] * J_zeta[:, 1, 1]
        )
        d2u[:, 1, 1] = (
            d2udzeta2[:, 0, 1] * J_zeta[:, 0, 0] * J_zeta[:, 0, 1]
            + d2udzeta2[:, 1, 1]
            * (J_zeta[:, 0, 0] * J_zeta[:, 1, 1] + J_zeta[:, 1, 0] * J_zeta[:, 0, 1])
            + d2udzeta2[:, 2, 1] * J_zeta[:, 1, 0] * J_zeta[:, 1, 1]
        )
        d2u[:, 2, 0] = (
            d2udzeta2[:, 0, 0] * J_zeta[:, 0, 1] ** 2
            + 2 * d2udzeta2[:, 1, 0] * J_zeta[:, 0, 1] * J_zeta[:, 1, 1]
            + d2udzeta2[:, 2, 0] * J_zeta[:, 1, 1] ** 2
        )
        d2u[:, 2, 1] = (
            d2udzeta2[:, 0, 1] * J_zeta[:, 0, 1] ** 2
            + 2 * d2udzeta2[:, 1, 1] * J_zeta[:, 0, 1] * J_zeta[:, 1, 1]
            + d2udzeta2[:, 2, 1] * J_zeta[:, 1, 1] ** 2
        )

        self._du = du
        self._d2u = d2u

    def _reliability_guided(self):
        """

        Private method to perform reliability-guided (RG) PIV analysis.

        """
        # Set up.
        m = np.shape(self._nodes)[0]
        n = np.shape(self._p_0)[0]
        self._subset_solved = np.zeros(
            m, dtype=int
        )  # Solved/unsolved reference array (1 if unsolved, -1 if solved).
        self._C_ZNCC = np.zeros(m, dtype=np.float64)  # Correlation coefficient array.
        self._subsets = np.empty(m, dtype=object)  # Initiate subset array.
        self._p = np.zeros((m, n), dtype=np.float64)  # Warp function array.
        self._displacements = np.zeros(
            (m, 2), dtype=np.float64
        )  # Displacement output array.

        # All nodes.
        entities = gmsh.model.getEntities()
        self._node_tags = []
        for e in entities:
            tags, _, _ = gmsh.model.mesh.getNodes(e[0], e[1])
            self._node_tags = np.append(self._node_tags, tags.flatten()).astype(int)

        # Interior and boundary nodes.
        entities = gmsh.model.getEntities(2)
        self._interior_node_tags = []
        for e in entities:
            tags, _, _ = gmsh.model.mesh.getNodes(e[0], e[1])
            self._interior_node_tags = np.append(
                self._interior_node_tags, tags.flatten()
            ).astype(int)
        self._borders_node_tags = (
            np.setdiff1d(self._node_tags, self._interior_node_tags).astype(int) - 1
        )

        # Template masking using binary mask.
        for tag in range(len(self._node_tags)):
            if tag in self._borders_node_tags:
                centre = self._nodes[tag]
                template = deepcopy(self._template)
                template.mask(centre, self._mask)
                self._subsets[tag] = gp.subset.Subset(
                    f_coord=self._nodes[tag],
                    f_img=self._f_img,
                    g_img=self._g_img,
                    template=template,
                )  # Create masked boundary subset.
            else:
                self._subsets[tag] = gp.subset.Subset(
                    f_coord=self._nodes[tag],
                    f_img=self._f_img,
                    g_img=self._g_img,
                    template=self._template,
                )  # Create full subset.
        # Solve subsets in mesh.
        number_nodes = np.shape(self._nodes)[0]
        with alive_bar(
            number_nodes, dual_line=True, bar="blocks", title=self._message
        ) as self._bar:
            # Solve for seed.
            self._bar.text = "-> Solving seed subset..."
            self._subsets[self._seed_node].solve(
                max_norm=self._max_norm,
                max_iterations=self._max_iterations,
                p_0=self._p_0,
                order=self._subset_order,
                method=self._method,
                tolerance=0.9,
            )  # Solve for seed subset.
            self._bar()

            # If seed not solved, log the error, otherwise store
            # the variables and solve neighbours.
            if not self._subsets[self._seed_node].data["solved"]:
                self._update = True
                log.error("Error! The seed subset correlation is below tolerance.")
            else:
                self._store_variables(self._seed_node, seed=True)

                # Solve for neighbours of the seed subset.
                p_0 = self._subsets[self._seed_node].data["results"][
                    "p"
                ]  # Set seed subset warp function as the preconditioning.
                self._neighbours(
                    self._seed_node, p_0
                )  # Solve for neighbouring subsets.

                # Solve through sorted queue.
                self._bar.text = (
                    "-> Solving remaining subsets using reliability guided approach..."
                )
                count = 1
                while np.max(self._subset_solved) > -1:
                    # Identify next subset.
                    cur_idx = np.argmax(
                        self._subset_solved * self._C_ZNCC
                    )  # Subset with highest correlation coefficient selected.
                    p_0 = self._subsets[cur_idx].data["results"][
                        "p"
                    ]  # Precondition based on selected subset.
                    self._subset_solved[cur_idx] = -1  # Set as solved.
                    solved = self._neighbours(
                        cur_idx, p_0
                    )  # Calculate for neighbouring subsets.
                    count += 1
                    self._bar()
                    if count == number_nodes:
                        break
                    elif solved is False:
                        break

        # Update check.
        if any(self._subset_solved != -1):
            self._update = True
            self.solved = False
            self._unsolvable = True
        else:
            # Compute element areas and strains.
            self._update = False
            self.solved = True
            self._unsolvable = False
            self._element_area()
            self._element_strains()
            self._update_subset_bgf()

    def _connectivity(self, idx):
        """

        A private method that returns the indices of nodes connected
        to the index node according to the input array.

        Parameters
        ----------
        idx : int
            Index of node.


        Returns
        -------
        pts_idx : `numpy.ndarray` (N)
            Mesh array.

        """
        element_idxs = np.argwhere(self._elements == idx)
        pts_idxs = []
        for i in range(len(element_idxs)):
            if element_idxs[i, 1] == 0:  # If 1
                pts_idxs.append(self._elements[element_idxs[i, 0], 3::2])  # Add 4,6
            elif element_idxs[i, 1] == 1:  # If 2
                pts_idxs.append(self._elements[element_idxs[i, 0], 3:5])  # Add 4,5
            elif element_idxs[i, 1] == 2:  # If 3
                pts_idxs.append(self._elements[element_idxs[i, 0], 4:])  # Add 5,6
            elif element_idxs[i, 1] == 3:  # If 4
                pts_idxs.append(self._elements[element_idxs[i, 0], :2])  # Add 1,2
            elif element_idxs[i, 1] == 4:  # If 5
                pts_idxs.append(self._elements[element_idxs[i, 0], 1:3])  # Add 2,3
            elif element_idxs[i, 1] == 5:  # If 6
                pts_idxs.append(self._elements[element_idxs[i, 0], :3:2])  # Add 1,3
        pts_idxs = np.unique(pts_idxs)

        return pts_idxs

    def _neighbours(self, cur_idx, p_0):
        """

        Private method to calculate the correlation coefficients and
        warp functions of the neighbouring nodes.

        Parameters
        __________
        p_0 : `numpy.ndarray` (N)
            Preconditioning warp function.


        Returns
        -------
        solved : bool
            Boolean to indicate whether the neighbouring subsets have been solved.

        """
        neighbours = self._connectivity(cur_idx)
        for idx in neighbours:
            if self._subset_solved[idx] == 0:  # If not previously solved.
                # Use nearest-neighbout pre-conditioning.
                self._subsets[idx].solve(
                    max_norm=self._max_norm,
                    max_iterations=self._max_iterations,
                    order=self._subset_order,
                    p_0=p_0,
                    method=self._method,
                    tolerance=self._tolerance,
                )
                if self._subsets[idx].data["solved"]:  # Check against tolerance.
                    self._store_variables(idx)
                else:
                    # Try more extrapolated pre-conditioning.
                    diff = self._nodes[idx] - self._nodes[cur_idx]
                    p = self._subsets[cur_idx].data["results"]["p"]
                    if np.shape(p)[0] == 6:
                        p_0[0] = p[0] + p[2] * diff[0] + p[3] * diff[1]
                        p_0[1] = p[1] + p[4] * diff[0] + p[5] * diff[1]
                    elif np.shape(p)[0] == 12:
                        p_0[0] = (
                            p[0]
                            + p[2] * diff[0]
                            + p[3] * diff[1]
                            + 0.5 * p[6] * diff[0] ** 2
                            + p[7] * diff[0] * diff[1]
                            + 0.5 * p[8] * diff[1] ** 2
                        )
                        p_0[1] = (
                            p[1]
                            + p[4] * diff[0]
                            + p[5] * diff[1]
                            + 0.5 * p[9] * diff[0] ** 2
                            + p[10] * diff[0] * diff[1]
                            + 0.5 * p[11] * diff[1] ** 2
                        )
                    self._subsets[idx].solve(
                        max_norm=self._max_norm,
                        max_iterations=self._max_iterations,
                        p_0=p_0,
                        method=self._method,
                        tolerance=self._tolerance,
                    )
                    if self._subsets[idx].data["solved"]:
                        self._store_variables(idx)
                    else:
                        # Finally, try the NCC initial guess.
                        self._subsets[idx].solve(
                            max_norm=self._max_norm,
                            max_iterations=self._max_iterations,
                            p_0=np.zeros(np.shape(p_0)),
                            method=self._method,
                            tolerance=self._tolerance,
                        )
                        if self._subsets[idx].data["solved"]:
                            self._store_variables(idx)
                            return True
                        else:
                            return False

    def _store_variables(self, idx, seed=False):
        """

        Private method to store variables.

        Parameters
        ----------
        idx : int
            Index of current subset.
        seed : bool
            Boolean to indicate whether this is the seed subset.

        """
        if seed is True:
            flag = -1
        else:
            flag = 1
        self._subset_solved[idx] = flag
        self._C_ZNCC[idx] = np.max(
            (self._subsets[idx].data["results"]["C_ZNCC"], 0)
        )  # Clip correlation coefficient to positive values.
        self._p[idx] = self._subsets[idx].data["results"]["p"].flatten()
        self._displacements[idx, 0] = self._subsets[idx].data["results"]["u"]
        self._displacements[idx, 1] = self._subsets[idx].data["results"]["v"]


class MeshResults(MeshBase):
    """

    MeshResults class for geopyv.

    Parameters
    ----------
    data : dict
        geopyv data dict from Mesh object.


    Attributes
    ----------
    data : dict
        geopyv data dict from Mesh object.

    """

    def __init__(self, data):
        """Initialisation of geopyv MeshResults class."""
        self.data = data
