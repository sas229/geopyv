"""

Mesh module for geopyv.

"""
import logging
import numpy as np
import geopyv as gp
import scipy as sp
from geopyv.object import Object
import gmsh
from copy import deepcopy
from scipy.optimize import minimize_scalar
from alive_progress import alive_bar
import traceback

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

    def inspect(self, *, subset_index=None, show=True, block=True, save=None):
        """

        Method to show the mesh and associated subset quality metrics using
        :mod: `~geopyv.plots.inspect_subset` for subsets and
        :mod: `~geopyv.plots.inspect_mesh` for meshes.

        Parameters
        ----------
        subset_index : int, optional
            Index of the subset to inspect. If `None', the mesh is inspected instead.
        show : bool, optional
            Control whether the plot is displayed.
        block : bool, optional
            Control whether the plot blocks execution until closed.
        save : str, optional
            Name to use to save plot. Uses default extension of `.png`.

        Returns
        -------
        fig :  `matplotlib.pyplot.figure`
            Figure object.
        ax : `matplotlib.pyplot.axes`
            Axes object.

        .. note::
            * The figure and axes objects can be returned allowing standard
              matplotlib functionality to be used to augment the plot generated.
              See the :ref:`plots tutorial <Plots Tutorial>` for guidance.

        .. seealso::
            :meth:`~geopyv.plots.inspect_subset`
            :meth:`~geopyv.plots.inspect_mesh`

        """

        # Check input.
        self._report(
            gp.check._check_type(subset_index, "subset_index", [int, type(None)]),
            "TypeError",
        )
        if subset_index:
            self._report(
                gp.check._check_index(
                    subset_index, "subset_index", 0, self.data["nodes"]
                ),
                "IndexError",
            )
        self._report(gp.check._check_type(show, "show", [bool]), "TypeError")
        self._report(gp.check._check_type(block, "block", [bool]), "TypeError")
        self._report(gp.check._check_type(save, "save", [str, type(None)]), "TypeError")

        # Inspect.
        if subset_index is not None:
            if self.data["solved"]:
                subset_data = self.data["results"]["subsets"][subset_index]
            else:
                subset_data = gp.subset.Subset(
                    f_coord=self._nodes[subset_index],
                    f_img=self._f_img,
                    g_img=self._g_img,
                    template=self._template,
                ).data
            mask = self.data["mask"]
            log.info("Inspecting subset {subset}...".format(subset=subset_index))
            fig, ax = gp.plots.inspect_subset(
                data=subset_data,
                mask=mask,
                show=show,
                block=block,
                save=save,
            )
            return fig, ax
        else:
            log.info("Inspecting mesh...")
            fig, ax = gp.plots.inspect_mesh(
                data=self.data,
                show=show,
                block=block,
                save=save,
            )
            return fig, ax

    def convergence(
        self,
        *,
        subset_index=None,
        quantity=None,
        show=True,
        block=True,
        save=None,
    ):
        """

        Method to plot the rate of convergence using
        :mod: `~geopyv.plots.convergence_subset` for subsets and
        :mod: `~geopyv.plots.convergence_mesh` for meshes.

        Parameters
        ----------
        subset_index : int, optional
            Index of the subset to inspect.
            If `None', the convergence plot is for the mesh instead.
        quantity : str, optional
            Selector for histogram convergence property if the convergence
            plot is for mesh. Defaults to `C_ZNCC` if left as default None.
        show : bool, optional
            Control whether the plot is displayed.
        block : bool, optional
            Control whether the plot blocks execution until closed.
        save : str, optional
            Name to use to save plot. Uses default extension of `.png`.


        Returns
        -------
        fig :  matplotlib.pyplot.figure
            Figure object.
        ax : `matplotlib.pyplot.axes`
            Axes object.


        .. note::
            * The figure and axes objects can be returned allowing standard
              matplotlib functionality to be used to augment the plot generated.
              See the :ref:`plots tutorial <Plots Tutorial>` for guidance.

        .. warning::
            * Can only be used once the mesh has been solved using the
              :meth:`~geopyv.mesh.Mesh.solve` method.

        .. seealso::
            :mod: `~geopyv.plots.convergence_subset`
            :meth:`~geopyv.plots.convergence_mesh`

        """

        # Check if solved.
        if self.data["solved"] is not True:
            log.error(
                "Mesh not yet solved therefore no convergence data to plot. "
                "First, run :meth:`~geopyv.mesh.Mesh.solve()` to solve."
            )
            raise ValueError(
                "Mesh not yet solved therefore no convergence data to plot. "
                "First, run :meth:`~geopyv.mesh.Mesh.solve()` to solve."
            )

        # Check input.
        self._report(
            gp.check._check_type(subset_index, "subset_index", [int, type(None)]),
            "TypeError",
        )
        if subset_index:
            self._report(
                gp.check._check_index(
                    subset_index, "subset_index", 0, self.data["nodes"]
                ),
                "IndexError",
            )
        self._report(
            gp.check._check_type(quantity, "quantity", [str, type(None)]), "TypeError"
        )
        if quantity:
            self._report(
                gp.check._check_value(
                    quantity, "quantity", ["C_ZNCC", "iterations", "norm"]
                ),
                "ValueError",
            )
        self._report(gp.check._check_type(show, "show", [bool]), "TypeError")
        self._report(gp.check._check_type(block, "block", [bool]), "TypeError")
        self._report(gp.check._check_type(save, "save", [str, type(None)]), "TypeError")

        # Plot convergence.
        if subset_index is not None:
            log.info(
                "Generating convergence plots for subset {subset}...".format(
                    subset=subset_index
                )
            )
            fig, ax = gp.plots.convergence_subset(
                self.data["results"]["subsets"][subset_index],
                show=show,
                block=block,
                save=save,
            )
            return fig, ax
        else:
            if quantity is None:
                quantity = "C_ZNCC"
            log.info(
                "Generating {quantity} convergence histogram for mesh...".format(
                    quantity=quantity
                )
            )
            fig, ax = gp.plots.convergence_mesh(
                data=self.data,
                quantity=quantity,
                show=show,
                block=block,
                save=save,
            )
            return fig, ax

    def contour(
        self,
        *,
        quantity="C_ZNCC",
        imshow=True,
        colorbar=True,
        ticks=None,
        mesh=False,
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

        Method to plot the contours of a given quantity.

        Parameters
        ----------
        quantity : str, optional
            Selector for contour parameter. Must be in:
            [`C_ZNCC`, `iterations`, `norm`, `u`, `v`, `u_x`, `v_x`,`u_y`,`v_y`, `R`]
            Defaults to `C_ZNCC`.
        imshow : bool, optional
            Control whether the reference image is plotted.
            Defaults to True.
        colorbar : bool, optional
            Control whether the colour bar is plotted.
            Defaults to True.
        ticks : list, optional
            Overwrite default colourbar ticks.
            Defaults to None.
        mesh : bool, optional
            Control whether the mesh is plotted.
            Defaults to False.
        alpha : float, optional
            Control contour opacity. Must be between 0.0-1.0.
            Defaults to 0.75.
        levels : int or array-like, optional
            Control number and position of contour lines.
            Defaults to None.
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


        Returns
        -------
        fig :  matplotlib.pyplot.figure
            Figure object.
        ax : `matplotlib.pyplot.axes`
            Axes object.


        .. note::
            * The figure and axes objects can be returned allowing standard
              matplotlib functionality to be used to augment the plot generated.
              See the :ref:`plots tutorial <Plots Tutorial>` for guidance.

        .. warning::
            * Can only be used once the mesh has been solved using the
              :meth:`~geopyv.mesh.Mesh.solve` method.

        .. seealso::
            :meth:`~geopyv.plots.contour_mesh`

        """
        # Check if solved.
        if self.data["solved"] is not True:
            log.error(
                "Mesh not yet solved therefore no convergence data to plot. "
                "First, run :meth:`~geopyv.mesh.Mesh.solve()` to solve."
            )
            raise ValueError(
                "Mesh not yet solved therefore no convergence data to plot. "
                "First, run :meth:`~geopyv.mesh.Mesh.solve()` to solve."
            )

        # Check input.
        self._report(
            gp.check._check_type(quantity, "quantity", [str, type(None)]), "TypeError"
        )
        types = [
            "C_ZNCC",
            "iterations",
            "norm",
            "u",
            "v",
            "u_x",
            "v_x",
            "u_y",
            "v_y",
            "R",
        ]
        if quantity:
            self._report(
                gp.check._check_value(quantity, "quantity", types), "ValueError"
            )
        self._report(gp.check._check_type(imshow, "imshow", [bool]), "TypeError")
        self._report(gp.check._check_type(colorbar, "colorbar", [bool]), "TypeError")
        types = [tuple, list, np.ndarray, type(None)]
        self._report(gp.check._check_type(ticks, "ticks", types), "TypeError")
        self._report(gp.check._check_type(mesh, "mesh", [bool]), "TypeError")
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

        # Plot contours.
        log.info(
            "Generating {quantity} contour plot for mesh...".format(quantity=quantity)
        )
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
        *,
        scale=1.0,
        imshow=True,
        mesh=False,
        axis=True,
        xlim=None,
        ylim=None,
        show=True,
        block=True,
        save=None,
    ):
        """

        Method to plot a quiver plot of the displacements.

        Parameters
        ----------
        scale : float, optional
            Control size of quiver arrows.
        imshow : bool, optional
            Control whether the reference image is plotted.
            Defaults to True.
        mesh : bool, optional
            Control whether the mesh is plotted.
            Defaults to False.
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

        Returns
        -------
        fig :  matplotlib.pyplot.figure
            Figure object.
        ax : `matplotlib.pyplot.axes`
            Axes object.


        .. note::
            * The figure and axes objects can be returned allowing standard
              matplotlib functionality to be used to augment the plot generated.
              See the :ref:`plots tutorial <Plots Tutorial>` for guidance.

        .. warning::
            * Can only be used once the mesh has been solved using the
              :meth:`~geopyv.mesh.Mesh.solve` method.

        .. seealso::
            :meth:`~geopyv.plots.quiver_mesh`
        """

        # Check if solved.
        if self.data["solved"] is not True:
            log.error(
                "Mesh not yet solved therefore no convergence data to plot. "
                "First, run :meth:`~geopyv.mesh.Mesh.solve()` to solve."
            )
            raise ValueError(
                "Mesh not yet solved therefore no convergence data to plot. "
                "First, run :meth:`~geopyv.mesh.Mesh.solve()` to solve."
            )

        # Check inputs.
        check = gp.check._check_type(scale, "scale", [float])
        if check:
            try:
                scale = float(scale)
                self._report(gp.check._conversion(scale, "scale", float), "Warning")
            except Exception:
                self._report(check, "TypeError")
        self._report(gp.check._check_range(scale, "scale", 0.0), "ValueError")
        self._report(gp.check._check_type(imshow, "imshow", [bool]), "TypeError")
        self._report(gp.check._check_type(mesh, "mesh", [bool]), "TypeError")
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

        # Plot quiver.
        log.info("Generating quiver plot for mesh...")
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


class Mesh(MeshBase):
    """
    Mesh class for geopyv.

    Private Attributes
    ------------------
    _f_img : `geopyv.image.Image`
        Reference image of geopyv.image.Image class, instantiated by
        :mod:`~image.Image`.
    _g_img : `geopyv.image.Image`
        Target image of geopyv.image.Image class, instantiated by
        :mod:`~image.Image`.
    _target_nodes : int
        Target number of nodes. Defaults to a value of 1000.
    _boundary_obj : `geopyv.geometry.region.Circle` or `geopyv.geometry.region.Path`
        geopyv.geometry.region.Region class defining the boundary geometry and
        behaviours.
    _exclusion_objs : list
        List of `geopyv.geometry.region.Region` class objects defining the
        exclusion geometry and behaviours.
    _size_lower_bound : float
        Lower bound on element size.
    _size_upper_bound : float
        Lower bound on element size.
    _mesh_order : int
        Mesh element order. Either 1 or 2.
    _unsolvable : bool
        Boolean to indicate if the mesh cannot be solved.
    _nodes : `numpy.ndarray` (Nx,2)
        Array of subset coordinates.
    _elements : `numpy.ndarray(Nx,3*_mesh_order)
        Node connectivity matrix.

    """

    def __init__(
        self,
        *,
        f_img=None,
        g_img=None,
        target_nodes=1000,
        boundary_obj=None,
        exclusion_objs=[],
        size_lower_bound=1.0,
        size_upper_bound=1000.0,
        mesh_order=2,
    ):
        """

        Initialisation of geopyv mesh object.

        Parameters
        ----------
        f_img : geopyv.image.Image, optional
            Reference image of geopyv.image.Image class, instantiated
            by :mod:`~geopyv.image.Image`.
        g_img : geopyv.image.Image, optional
            Target image of geopyv.imageImage class, instantiated by
            :mod:`~geopyv.image.Image`.
        target_nodes : int, optional
            Target number of nodes. Defaults to a value of 1000.
        boundary_obj : geopyv.geometry.region.Region
            Boundary of geopyv.geometry.region.Region class, instantiated
            by :mod: `~geopyv.geometry.region.Region` to define the boundary
            of the region of interest (RoI).
        exclusion_objs : list, optional
            List of geopyv.geometry.region.Region class objects, instantiated
            by :mod: `~geopyv.geometry.region.Region` to define exclusion regions
            within the region of interest (RoI).
        size_lower_bound : float, optional
            Lower bound on element size. Defaults to a value of 1.0.
        size_upper_bound : float, optional
            Lower bound on element size. Defaults to a value of 1000.0.
        mesh_order : int, optional
            Mesh element order. Options are 1 and 2. Defaults to 2.

        Attributes
        ----------
        data : dict
            Data object containing all settings and results.
            See the data structure :ref:`here <mesh_data_structure>`.
        solved : bool
            Boolean to indicate if the mesh has been solved.

        """

        # Set initialised boolean.
        self.initialised = False

        # Check types.
        if self._report(
            gp.check._check_type(f_img, "f_img", [gp.image.Image]), "Warning"
        ):
            f_img = gp.io._load_f_img()
        if self._report(
            gp.check._check_type(g_img, "g_img", [gp.image.Image]), "Warning"
        ):
            g_img = gp.io._load_g_img()
        check = gp.check._check_type(target_nodes, "target_nodes", [int])
        if check:
            try:
                target_nodes = int(target_nodes)
                self._report(
                    gp.check._conversion(target_nodes, "target_nodes", int), "Warning"
                )
            except Exception:
                self._report(check, "TypeError")
        self._report(
            gp.check._check_range(target_nodes, "target_nodes", 1), "ValueError"
        )
        self._report(
            gp.check._check_type(
                boundary_obj,
                "boundary_obj",
                [gp.geometry.region.Circle, gp.geometry.region.Path],
            ),
            "TypeError",
        )
        check = gp.check._check_type(exclusion_objs, "exclusion_objs", [list])
        if check:
            try:
                exclusion_objs = list(exclusion_objs)
                self._report(
                    gp.check._conversion(exclusion_objs, "exclusion_objs", list, False),
                    "Warning",
                )
            except Exception:
                self._report(check, "TypeError")
        for exclusion_obj in exclusion_objs:
            self._report(
                gp.check._check_type(
                    exclusion_obj,
                    "exclusion_obj",
                    [gp.geometry.region.Circle, gp.geometry.region.Path],
                ),
                "TypeError",
            )
        self._report(
            gp.check._check_type(size_lower_bound, "size_lower_bound", [int, float]),
            "TypeError",
        )
        self._report(
            gp.check._check_type(size_upper_bound, "size_upper_bound", [int, float]),
            "TypeError",
        )
        self._report(
            gp.check._check_comp(
                size_lower_bound,
                "size_lower_bound",
                size_upper_bound,
                "size_upper_bound",
            ),
            "ValueError",
        )
        check = gp.check._check_type(mesh_order, "mesh_order", [int])
        if check:
            try:
                mesh_order = int(mesh_order)
                self._report(
                    gp.check._conversion(mesh_order, "mesh_order", int), "Warning"
                )
            except Exception:
                self._report(check, "TypeError")
        self._report(
            gp.check._check_value(mesh_order, "mesh_order", [1, 2]), "ValueError"
        )

        # Store.
        self._initialised = False
        self._f_img = f_img
        self._g_img = g_img
        self._target_nodes = target_nodes
        self._boundary_obj = boundary_obj
        self._boundary_obj._rigid = False
        self._exclusion_objs = exclusion_objs
        self._size_lower_bound = size_lower_bound
        self._size_upper_bound = size_upper_bound
        self._mesh_order = mesh_order
        self._hard_boundary = boundary_obj._hard
        self.solved = False
        self._unsolvable = False

        # Checks.
        self._target_checks()  # Size bounds and target nodes.
        self._update_region()  # Boundary and exclusion objects.

        # Creating the initial mesh.
        # Define region of interest.
        (
            self._borders,
            self._segments,
            self._curves,
            self._mask,
        ) = gp.geometry.meshing._define_RoI(
            self._f_img, self._boundary_obj, self._exclusion_objs
        )
        # Initialize gmsh if not already initialized.
        if gmsh.isInitialized() == 0:
            gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)
        # 0: silent except for fatal errors, 1: +errors, 2: +warnings,
        # 3: +direct, 4: +information, 5: +status, 99: +debug.
        log.info(
            "Generating mesh using gmsh with approximately {n} nodes.".format(
                n=self._target_nodes
            )
        )
        self._initial_mesh()
        self._update_mesh()
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
            "mesh_order": self._mesh_order,
            "boundary_obj": self._boundary_obj,
            "exclusion_objs": self._exclusion_objs,
            "boundary": self._boundary,
            "exclusions": self._exclusions,
            "size_lower_bound": self._size_lower_bound,
            "size_upper_bound": self._size_upper_bound,
            "nodes": self._nodes,
            "elements": self._elements,
            "mask": self._mask,
        }

        self.initialised = True

    def _target_checks(self):
        """Private method to limit the mesh size upper bound by the boundary geometry
        and the target nodes by the number included in the boundary and exclusion
        geometries."""

        self._size_upper_bound = min(
            self._size_upper_bound,
            np.max(
                np.sqrt(
                    np.sum(
                        np.square(np.diff(self._boundary_obj._nodes, axis=0)), axis=1
                    )
                )
            ),
        )
        minimum_nodes = np.shape(self._boundary_obj._nodes)[0]
        for exclusion_obj in self._exclusion_objs:
            minimum_nodes += np.shape(exclusion_obj._nodes)[0]
        self._target_nodes = max(self._target_nodes, minimum_nodes)

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

        # Check inputs.
        check = gp.check._check_type(target_nodes, "target_nodes", [int])
        if check:
            try:
                target_nodes = int(target_nodes)
                self._report(
                    gp.check._conversion(target_nodes, "target_nodes", int), "Warning"
                )
            except Exception:
                self._report(check, "TypeError")

        # Store.
        self._target_nodes = target_nodes

        # Checks.
        self._target_checks()  # Size bounds and target nodes.

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
        self._update_mesh()
        log.info(
            "Mesh generated with {n} nodes and {e} elements.".format(
                n=len(self._nodes), e=len(self._elements)
            )
        )
        gmsh.finalize()

    def solve(
        self,
        *,
        template=None,
        seed_coord=None,
        seed_warp=None,
        max_norm=1e-5,
        max_iterations=50,
        subset_order=1,
        tolerance=0.75,
        seed_tolerance=0.9,
        method="ICGN",
        adaptive_iterations=0,
        correction=True,
        alpha=0.5,
    ):
        r"""

        Method to solve for the mesh.

        Parameters
        ----------
        template : `geopyv.templates.Template`
            Subset template object.
        seed_coord : `numpy.ndarray` (2,)
            An image coordinate selected in a region of low deformation. The
            reliability-guided approach is initiated from the nearest subset.
        seed_warp : `numpy.ndarray` (6*subset_order,)
            Deformation preconditioning vector for the seed subset.
        max_norm : float, optional
            Exit criterion for norm of increment in warp function.
            Defaults to value of
            :math:`1 \cdot 10^{-5}`.
        max_iterations : int, optional
            Exit criterion for number of Gauss-Newton iterations.
            Defaults to value
            of 50.
        subset_order : int
            Warp function order. Options are 1 and 2.
            Defaults to a value of 1.
        tolerance: float, optional
            Correlation coefficient tolerance.
            Defaults to a value of 0.75.
        seed_tolerance: float, optional
            Correlation coefficient tolerance for the seed subset.
            Defaults to a value of 0.9.
        method : str
            Solution method. Options are FAGN and ICGN.
            Default is ICGN since it is faster.
        adaptive_iterations : int, optional
            Number of mesh adaptivity iterations to perform.
            Defaults to a value of 0.
        correction : bool, optional
            Boolean indicator as to whether to correct poor
            correlation subsets.
            Defaults to a value of True.
        alpha : float, optional
            Mesh adaptivity control parameter.
            Defaults to a value of 0.5.

        Returns
        -------
        solved : bool
            Boolean to indicate if the mesh instance has been solved.

        .. note::
            * For guidance on how to use this class see the
              :ref:`mesh tutorial <Mesh Tutorial>`.

        """

        # Check if solved.
        if self.data["solved"] is True:
            log.error("Mesh has already been solved. Cannot be solved again.")
            return

        # Check inputs.
        if template is None:
            template = gp.templates.Circle(50)
        types = [gp.templates.Circle, gp.templates.Square]
        self._report(gp.check._check_type(template, "template", types), "TypeError")
        if self._report(
            gp.check._check_type(seed_coord, "seed_coord", [np.ndarray]), "Warning"
        ):
            selector = gp.gui.selectors.coordinate.CoordinateSelector()
            seed_coord = selector.select(self._f_img, template)
        elif self._report(gp.check._check_dim(seed_coord, "seed_coord", 1), "Warning"):
            selector = gp.gui.selectors.coordinate.CoordinateSelector()
            seed_coord = selector.select(self._f_img, template)
        elif self._report(
            gp.check._check_axis(seed_coord, "seed_coord", 0, [2]), "Warning"
        ):
            selector = gp.gui.selectors.coordinate.CoordinateSelector()
            seed_coord = selector.select(self._f_img, template)
        if seed_warp is not None:
            check = gp.check._check_type(seed_warp, "seed_warp", [np.ndarray])
            if check:
                try:
                    seed_warp = np.asarray(seed_warp)
                    self._report(
                        gp.check._conversion(seed_warp, "seed_warp", np.ndarray),
                        "Warning",
                    )
                except Exception:
                    self._report(check, "TypeError")
            self._report(gp.check._check_dim(seed_warp, "seed_warp", 1), "ValueError")
            self._report(
                gp.check._check_axis(seed_warp, "seed_warp", 0, [6 * subset_order]),
                "ValueError",
            )
        else:
            seed_warp = np.zeros(6 * subset_order)
        check = gp.check._check_type(max_norm, "max_norm", [float])
        if check:
            try:
                max_norm = float(max_norm)
                self._report(
                    gp.check._conversion(max_norm, "max_norm", float), "Warning"
                )
            except Exception:
                self._report(check, "TypeError")
        self._report(gp.check._check_range(max_norm, "max_norm", 1e-20), "ValueError")
        check = gp.check._check_type(max_iterations, "max_iterations", [int])
        if check:
            try:
                max_iterations = int(max_iterations)
                self._report(
                    gp.check._conversion(max_iterations, "max_iterations", int),
                    "Warning",
                )
            except Exception:
                self._report(check, "TypeError")
        self._report(
            gp.check._check_range(max_iterations, "max_iterations", 0.5), "ValueError"
        )
        check = gp.check._check_type(subset_order, "subset_order", [int])
        if check:
            try:
                subset_order = int(subset_order)
                self._report(
                    gp.check._conversion(subset_order, "subset_order", int), "Warning"
                )
            except Exception:
                self._report(check, "TypeError")
        self._report(
            gp.check._check_value(subset_order, "subset_order", [1, 2]), "ValueError"
        )
        check = gp.check._check_type(adaptive_iterations, "adaptive_iterations", [int])
        if check:
            try:
                adaptive_iterations = int(adaptive_iterations)
                self._report(
                    gp.check._conversion(
                        adaptive_iterations, "adaptive_iterations", int
                    ),
                    "Warning",
                )
            except Exception:
                self._report(check, "TypeError")
        self._report(
            gp.check._check_range(adaptive_iterations, "adaptive_iterations", 0),
            "ValueError",
        )
        check = gp.check._check_type(correction, "correction", [bool])
        if check:
            try:
                correction = bool(correction)
                self._report(
                    gp.check._conversion(correction, "correction", int),
                    "Warning",
                )
            except Exception:
                self._report(check, "TypeError")
        self._report(
            gp.check._check_range(correction, "correction", 0),
            "ValueError",
        )
        check = gp.check._check_type(tolerance, "tolerance", [float])
        if check:
            try:
                tolerance = float(tolerance)
                self._report(
                    gp.check._conversion(tolerance, "tolerance", float), "Warning"
                )
            except Exception:
                self._report(check, "TypeError")
        self._report(
            gp.check._check_range(tolerance, "tolerance", 0.0, 1.0), "ValueError"
        )
        check = gp.check._check_type(seed_tolerance, "seed_tolerance", [float])
        if check:
            try:
                seed_tolerance = float(seed_tolerance)
                self._report(
                    gp.check._conversion(seed_tolerance, "seed_tolerance", float),
                    "Warning",
                )
            except Exception:
                self._report(check, "TypeError")
        self._report(
            gp.check._check_range(seed_tolerance, "seed_tolerance", 0.0, 1.0),
            "ValueError",
        )
        self._report(
            gp.check._check_comp(
                tolerance,
                "tolerance",
                seed_tolerance,
                "seed_tolerance",
            ),
            "ValueError",
        )
        self._report(gp.check._check_type(method, "method", [str]), "TypeError")
        self._report(
            gp.check._check_value(method, "method", ["FAGN", "ICGN"]), "ValueError"
        )
        check = gp.check._check_type(alpha, "alpha", [float])
        if check:
            try:
                alpha = float(alpha)
                self._report(gp.check._conversion(alpha, "alpha", float), "Warning")
            except Exception:
                self._report(check, "TypeError")
        self._report(gp.check._check_range(alpha, "alpha", 0.0, 1.0), "ValueError")

        # Store variables.
        self._seed_coord = seed_coord
        self._template = template
        self._max_iterations = max_iterations
        self._max_norm = max_norm
        self._adaptive_iterations = adaptive_iterations
        self._correction = correction
        self._method = method
        self._subset_order = subset_order
        self._tolerance = tolerance
        self._seed_tolerance = seed_tolerance
        self._alpha = alpha
        self._update = False
        self._seed_warp = seed_warp
        self._status = 0

        # Initialize gmsh if not already initialized.
        if gmsh.isInitialized() == 0:
            gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)

        # Create initial mesh.
        self._initial_mesh()
        self._update_mesh()

        # Solve initial mesh.
        self._message = "Solving initial mesh"
        self._find_seed_node()
        try:
            self._reliability_guided()
            if self._unsolvable:
                self._check_status()
                return self.solved
            # Solve adaptive iterations.
            for iteration in range(1, adaptive_iterations + 1):
                self._message = "Adaptive iteration {}".format(iteration)
                self._adaptive_mesh()
                self._update_mesh()
                self._find_seed_node()
                self._reliability_guided()
                if self._unsolvable:
                    self._check_status()
                    return self.solved
            self._store_region()
            if self._unsolvable:
                self._check_status()
                return self.solved

            self._status = 1
            self._check_status()

            # Pack data.
            self.solved = True
            self.data["nodes"] = self._nodes
            self.data["elements"] = self._elements
            self.data["areas"] = self._areas
            self.data["solved"] = self.solved
            self.data["unsolvable"] = self._unsolvable
            self.data.update(
                {"centroids": np.mean(self._nodes[self._elements], axis=-2)}
            )

            # Pack settings.
            self._settings = {
                "max_iterations": self._max_iterations,
                "max_norm": self._max_norm,
                "adaptive_iterations": self._adaptive_iterations,
                "method": self._method,
                "tolerance": self._tolerance,
                "correction": self._correction,
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
                "warps": self._warps,
                "C_ZNCC": self._C_ZNCC,
                "seed": self._seed_node,
            }
            self.data.update({"results": self._results})

        except Exception:
            self._status = 7
            self._unsolvable = True
            self._check_status()
            print(traceback.format_exc())
        gmsh.finalize()
        return self.solved

    def _check_status(self):
        """Private method for issue catergorisation and error message delivery."""
        if self._status == 0:  # Solvable unsolved.
            return
        elif self._status == 1:  # Solvable solved.
            log.info(
                "Solved mesh. Minimum correlation coefficient: {min_C:.2f}; "
                "maximum correlation coefficient: {max_C:.2f}.".format(
                    min_C=np.amin(self._C_ZNCC),
                    max_C=np.amax(self._C_ZNCC),
                )
            )
        elif self._status == 2:  # Unsolvable: seed decorrelation.
            log.error(
                "Specified seed correlation coefficient tolerance not met."
                "Minimum seed correlation coefficient:"
                "{seed_C:.2f}; tolerance: {seed_tolerance:.2f}.".format(
                    seed_C=self._subsets[self._seed_node].data["results"]["C_ZNCC"],
                    seed_tolerance=self._seed_tolerance,
                )
            )
        elif self._status == 3:  # Unsolvable: subset decorrelation.
            log.error(
                "Specified correlation coefficient tolerance not met. "
                "Minimum correlation coefficient: "
                "{min_C:.2f}; tolerance: {tolerance:.2f}.".format(
                    min_C=np.amin(self._C_ZNCC[np.where(self._C_ZNCC > 0.0)]),
                    tolerance=self._tolerance,
                )
            )
        elif self._status == 4:  # Unsolvable: rigid exclusions decorrelation.
            log.error("Rigid exclusion tracking failure.")
        elif self._status == 5:  # Unsolvable: unsolvable subset.
            log.error("Unsolvable subset.")
        elif self._status == 6:  # Unsolvable: subset compatability failure.
            log.error("Local mesh compatibility failure.")
        elif self._status == 7:  # Unsolvable: unkown issue.
            log.error("Could not solve mesh. Unrecognised problem.")
        self._update = True

    def _store_region(self):
        """Private method to record the deformation of the boundary and exclusions."""
        exclusion_obj_warp = []
        for i in range(np.shape(self._exclusion_objs)[0]):
            if self._exclusion_objs[i]._rigid:  # Track if rigid.
                subset = gp.subset.Subset(
                    f_img=self._f_img,
                    g_img=self._g_img,
                    f_coord=self._exclusion_objs[i]._centre,
                    template=gp.templates.Circle(
                        int(self._exclusion_objs[i]._specifics["radius"])
                    ),
                )
                warp_0 = np.zeros(6 * self._subset_order)
                warp_0[:2] = np.mean(self._displacements[self._exclusions[i]], axis=0)
                subset.solve(tolerance=0.9, warp_0=warp_0, order=self._subset_order)
                if subset.data["solved"] is not True:
                    self._status = 4
                    del subset
                    return
                else:
                    exclusion_obj_warp.append(subset._p.flatten())
                    del subset
            else:
                exclusion_obj_warp.append(self._displacements[self._exclusions[i]])
        self._boundary_obj._store(self._displacements[self._boundary])
        for i in range(np.shape(self._exclusion_objs)[0]):
            self._exclusion_objs[i]._store(exclusion_obj_warp[i])

    def _update_region(self):
        """

        Private method to trigger region objects (boundary and exclusions) to update.

        """

        self._boundary_obj._update(self._f_img.filepath)
        for exclusion_obj in self._exclusion_objs:
            exclusion_obj._update(self._f_img.filepath)

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
            (np.asarray(ent) - 1).flatten(), (-1, 3 * self._mesh_order)
        )  # Element connectivity array.
        self._boundary = gmsh.model.occ.getCurveLoops(0)[1][0]
        self._edges = list(gmsh.model.mesh.getNodesForPhysicalGroup(1, 0)[0] - 1)
        self._exclusions = []
        for i in range(len(self._exclusion_objs)):
            self._exclusions.append(np.sort(gmsh.model.occ.getCurveLoops(0)[1][i + 1]))
            self._edges += list(
                gmsh.model.mesh.getNodesForPhysicalGroup(1, i + 1)[0] - 1
            )

    def _find_seed_node(self):
        """

        Private method to find seed node given seed coordinate.

        """
        dist = np.sqrt(
            (self._nodes[:, 0] - self._seed_coord[0]) ** 2
            + (self._nodes[:, 1] - self._seed_coord[1]) ** 2
        )
        self._seed_node = np.argmin(dist)

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
            f,
            bounds=(self._size_lower_bound, self._size_upper_bound),
            method="bounded",
        )

    def _adaptive_mesh(self):
        """

        Private method to perform adaptive remeshing.

        """
        message = "Adaptively remeshing..."
        with alive_bar(dual_line=True, bar=None, title=message) as bar:
            D = (
                abs(self._warps[:, 3] + self._warps[:, 4]) * self._areas
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

            minimize_scalar(f, bounds=(0.1, 5))
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
        error = (np.shape(nodes)[0] - target_nodes) ** 2

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
            nodes[elements[:, :3]],
            ((0, 0), (0, 0), (0, 2)),
            mode="constant",
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

    def _element_area(self):
        """
        Private method to calculate the element areas based on corner nodes.

        """
        M = np.ones((len(self._elements), 3, 3))
        M[:, 1] = self._nodes[self._elements[:, :3]][:, :, 0]
        M[:, 2] = self._nodes[self._elements[:, :3]][:, :, 1]
        self._areas = 0.5 * np.linalg.det(M)

    def _local_coordinates(self):
        """
        Private method to define the element centroid in terms of the local
        coordinate system. Thereotically, this is constant but is calculated in
        case of rounding error.

        Return
        ------
        A : `numpy.ndarray` (N,3,4)
            Element centroid local coordinates.
        """

        A = np.ones((np.shape(self._elements)[0], 3, 4))
        A[:, 1:, 0] = np.mean(self._nodes[self._elements], axis=1)
        A[:, 1:, 1:] = self._nodes[self._elements][:, :3, :2].transpose(0, 2, 1)

        return A

    def _shape_function(self):
        """Private method for defining the shape functions at the
        element centroid.

        Return
        ------
        N : `numpy.ndarray`
            Shape function.
        dN : `numpy.ndarray`
            Shape function 1st order derivative.
        d2N : `numpy.ndarray`
            Shape function 2nd order derivative.
        """

        if self._mesh_order == 1:
            N = np.asarray([1 / 3, 1 / 3, 1 / 3])
            dN = np.asarray([[1, 0, -1], [0, 1, -1]])
            d2N = None
        elif self._mesh_order == 2:
            N = np.asarray([-1 / 9, -1 / 9, -1 / 9, 4 / 9, 4 / 9, 4 / 9])
            dN = np.asarray(
                [
                    [1 / 3, 0, -1 / 3, 4 / 3, -4 / 3, 0],
                    [0, 1 / 3, -1 / 3, 4 / 3, 0, -4 / 3],
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

    def _element_strains(self):
        """

        Private method to calculate the elemental strain the "B" matrix
        relating element node displacements to elemental strain.

        """

        # Setup.
        self._warps = np.zeros((np.shape(self._elements)[0], 12))
        x = self._nodes[self._elements]
        u = self._displacements[self._elements]

        # Coordinate matrix.
        A = self._local_coordinates()

        # Shape function and derivatives.
        N, dN, d2N = self._shape_function()

        # Displacements.
        self._warps[:, :2] = N @ u

        # 1st Order Strains
        J_x_T = dN @ x
        J_u_T = dN @ u

        self._warps[:, 2:6] = (np.linalg.inv(J_x_T) @ J_u_T).reshape(
            np.shape(self._elements)[0], -1
        )

        # 2nd Order Strains
        if self._mesh_order == 2:
            K_u = d2N @ u
            dz = np.zeros((np.shape(self._elements)[0], 2, 2))
            dz[:, 0, 0] = x[:, 1, 1] - x[:, 2, 1]
            dz[:, 0, 1] = x[:, 2, 1] - x[:, 0, 1]
            dz[:, 1, 0] = x[:, 2, 0] - x[:, 1, 0]
            dz[:, 1, 1] = x[:, 0, 0] - x[:, 2, 0]
            dz /= np.linalg.det(A[:, :, [1, 2, 3]])[:, None, None]

            K_x_inv = np.zeros((np.shape(self._elements)[0], 3, 3))
            K_x_inv[:, 0, 0] = dz[:, 0, 0] ** 2
            K_x_inv[:, 0, 1] = 2 * dz[:, 0, 0] * dz[:, 0, 1]
            K_x_inv[:, 0, 2] = dz[:, 0, 1] ** 2
            K_x_inv[:, 1, 0] = dz[:, 0, 0] * dz[:, 1, 0]
            K_x_inv[:, 1, 1] = dz[:, 0, 0] * dz[:, 1, 1] + dz[:, 0, 1] * dz[:, 1, 0]
            K_x_inv[:, 1, 2] = dz[:, 0, 1] * dz[:, 1, 1]
            K_x_inv[:, 2, 0] = dz[:, 1, 0] ** 2
            K_x_inv[:, 2, 1] = 2 * dz[:, 1, 0] * dz[:, 1, 1]
            K_x_inv[:, 2, 2] = dz[:, 1, 1] ** 2

            self._warps[:, 6:] = (K_x_inv @ K_u).reshape(
                np.shape(self._elements)[0], -1
            )

    def _seed_solve(self):
        """

        Private method for the solving of the seed node.

        """

        self._subsets[self._seed_node].solve(
            max_norm=self._max_norm,
            max_iterations=self._max_iterations,
            warp_0=self._seed_warp,
            order=self._subset_order,
            method=self._method,
            tolerance=self._seed_tolerance,
        )  # Solve for seed subset.
        if not self._subsets[self._seed_node].data["solved"]:  # If seed unsolved.
            self._subsets[self._seed_node].solve(
                max_norm=self._max_norm,
                max_iterations=self._max_iterations,
                warp_0=np.zeros(np.shape(self._seed_warp)),
                order=self._subset_order,
                method=self._method,
                tolerance=self._seed_tolerance,
            )
            if not self._subsets[self._seed_node].data["solved"]:  # If seed unsolved.
                self._unsolvable = True
                self._status = 2
                return
        # Store if solved.
        self._store_variables(idx=self._seed_node, flag=-1)

    def _reliability_guided(self):
        """

        Private method to perform reliability-guided (RG) PIV analysis.

        """

        # Set up.
        m = np.shape(self._nodes)[0]
        n = np.shape(self._seed_warp)[0]
        self._subset_solved = np.zeros(
            m, dtype=int
        )  # Solved/unsolved reference array (1 if unsolved, -1 if solved).
        self._C_ZNCC = np.zeros(m, dtype=np.float64)  # Correlation coefficient array.
        self._subsets = np.empty(m, dtype=object)  # Initiate subset array.
        self._p = np.zeros((m, n), dtype=np.float64)  # Warp function array.
        self._displacements = np.zeros(
            (m, 2), dtype=np.float64
        )  # Displacement output array.

        # Subset instantiation with template masking.
        self._subset_instantiation()

        # Solve subsets in mesh.
        with alive_bar(
            np.shape(self._nodes)[0],
            dual_line=True,
            bar="blocks",
            title=self._message,
        ) as self._bar:
            # Solve for seed.
            self._bar.text = "-> Solving seed subset..."
            self._seed_solve()
            if self._unsolvable:
                return
            self._bar()

            # Solve for seed neighbours.
            self._neighbours(
                self._seed_node, self._subsets[self._seed_node].data["results"]["p"]
            )
            if self._unsolvable:
                return

            # Solve through sorted queue.
            self._bar.text = (
                "-> Solving remaining subsets using reliability guided approach..."
            )
            while np.max(self._subset_solved) > -1:
                # Highest correlation subset selected.
                cur_idx = np.argmax(self._subset_solved * self._C_ZNCC)
                self._subset_solved[cur_idx] = -1  # Set as solved.
                p_0 = self._subsets[cur_idx].data["results"]["p"]  # Precondition.
                self._neighbours(cur_idx, p_0)  # Solve for neighbours.
                self._bar()
                if self._bar.current == np.shape(self._nodes)[0]:
                    break
                if self._unsolvable:
                    return

        # Corrections, calculations and checks.
        if self._correction:
            self._corrections()
        # self._compatibility()
        if self._unsolvable:
            return
        self._element_area()
        self._element_strains()
        if any(self._subset_solved != -1) or any(self._C_ZNCC < self._tolerance):
            self._unsolvable = True
            self._status = 3

    def _subset_instantiation(self):
        """Private method that instantiates the mesh subsets with masking."""

        for i in range(np.shape(self._nodes)[0]):
            if i in self._edges:
                template = deepcopy(self._template)
                template.mask(self._nodes[i], self._mask)
                check = False
                for e in range(len(self._exclusions)):
                    if i in self._exclusions[e]:
                        if self._exclusion_objs[e]._compensate:
                            check = True
                            break
                if (
                    i in self._boundary and self._boundary_obj._compensate
                ) or check is True:
                    if template.n_px < self._template.n_px:
                        size = int(
                            self._template.size
                            / np.sqrt(template.n_px / self._template.n_px)
                        )
                        if self._template.shape == "circle":
                            template = gp.templates.Circle(radius=size)
                        elif self._template.shape == "square":
                            template = gp.templates.Square(length=size)
                template.mask(self._nodes[i], self._mask)
                self._subsets[i] = gp.subset.Subset(
                    f_coord=self._nodes[i],
                    f_img=self._f_img,
                    g_img=self._g_img,
                    template=template,
                )  # Create masked boundary subset.
            else:
                self._subsets[i] = gp.subset.Subset(
                    f_coord=self._nodes[i],
                    f_img=self._f_img,
                    g_img=self._g_img,
                    template=self._template,
                )  # Create full subset.

    def _compatibility(self):
        """

        Private method to ensure compatibility (that the displacement of a subset
        nowhere exceeds the displacement bounds of neighbouring subsets).

        """
        for i in range(np.shape(self._subsets)[0]):
            if i not in self._edges:
                neighbours = self._connectivity(i, full=True)
                if len(neighbours) > 2:
                    hull = sp.spatial.Delaunay(
                        self._nodes[neighbours] + self._displacements[neighbours]
                    )
                    if hull.find_simplex(self._nodes[i] + self._displacements[i]) < 0:
                        fixed = self._compatability_correction(i, neighbours, hull)
                        if fixed is not True:
                            self._unsolvable = True
                            self._status = 6

    def _compatability_correction(self, idx, neighbours, hull):
        """

        Private method to correct subset displacement in the case of incompatibility.

        """

        subset = gp.subset.Subset(
            f_coord=self._nodes[idx],
            f_img=self._f_img,
            g_img=self._g_img,
            template=self._subsets[idx]._template,
        )
        warp = np.mean(
            self._p[neighbours[self._C_ZNCC[neighbours] > self._tolerance]],
            axis=0,
        )
        subset.solve(
            max_norm=self._max_norm,
            max_iterations=self._max_iterations,
            warp_0=warp,
            order=self._subset_order,
            method=self._method,
            tolerance=self._seed_tolerance,
        )
        coord = self._nodes[idx] + subset._p[:2].flatten()
        if hull.find_simplex(coord) >= 0 and subset.solved:
            self._subsets[idx] = subset
            self._store_variables(idx=idx, flag=-1)
            return True
        else:
            return False

    def _corrections(self):
        """

        Private method to improve subset correlation through neighbour
        preconditioning.

        """

        arg = np.argsort(self._C_ZNCC)
        corarg = arg[self._C_ZNCC[arg] < self._seed_tolerance]
        j = 0
        k = 0
        benefit = np.zeros(len(self._C_ZNCC))
        if np.shape(corarg)[0] >= 1:
            with alive_bar(
                np.shape(corarg)[0],
                dual_line=True,
                bar="blocks",
                title="Corrections",
            ) as bar:
                for idx in corarg:
                    k += 1
                    subset = gp.subset.Subset(
                        f_coord=self._nodes[idx],
                        f_img=self._f_img,
                        g_img=self._g_img,
                        template=self._subsets[idx]._template,
                    )
                    neighbours = self._connectivity(idx)
                    warp = np.mean(
                        self._p[neighbours[self._C_ZNCC[neighbours] > self._tolerance]],
                        axis=0,
                    )
                    subset.solve(
                        max_norm=self._max_norm,
                        max_iterations=self._max_iterations,
                        warp_0=warp,
                        order=self._subset_order,
                        method=self._method,
                        tolerance=self._seed_tolerance,
                    )
                    try:
                        if (
                            subset._C_ZNCC > self._subsets[idx]._C_ZNCC
                            and subset.solved
                        ):
                            j += 1
                            benefit[idx] = subset._C_ZNCC - self._subsets[idx]._C_ZNCC
                            self._subsets[idx] = subset
                            self._store_variables(idx=idx, flag=-1)
                    except Exception:
                        pass
                    bar()
                    bar.text = "{} corrections accepted".format(round(j / k, 3))
            log.info(
                (
                    "{ca} corrections accepted;"
                    "{max} maximum improvement;"
                    "{mean} mean improvement"
                ).format(
                    ca=round(j / k, 3),
                    max=round(np.max(benefit), 3),
                    mean=round(np.mean(benefit), 3),
                )
            )

    def _connectivity(self, idx, full=False):
        """

        A private method that returns the indices of nodes connected
        to the index node according to the input array.

        Parameters
        ----------
        idx : int
            Index of node.
        full : bool, optional
            False - return immediate nodes to the index.
            True - return all nodes sharing an element with the index.


        Returns
        -------
        pts_idx : `numpy.ndarray` (N)
            Mesh array.

        """

        if self._mesh_order == 1 or full is True:
            element_idxs = np.argwhere(np.any(self._elements == idx, axis=1)).flatten()
            pts_idxs = np.unique(self._elements[element_idxs])
            pts_idxs = np.delete(pts_idxs, np.argwhere(pts_idxs == idx))
        elif self._mesh_order == 2:
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
        ----------
        cur_idx : int
            The current susbet index.
        p_0 : `numpy.ndarray` (N)
            Preconditioning warp function.


        Returns
        -------
        solved : bool
            Boolean to indicate whether the neighbouring subsets have been solved.

        """

        # Identify neighbours.
        neighbours = self._connectivity(cur_idx)

        # Iterate through list of neighbours.
        for idx in neighbours:
            if self._subset_solved[idx] == 0:  # If not previously solved.
                # 1. Use nearest-neighbour pre-conditioning.
                self._subsets[idx].solve(
                    max_norm=self._max_norm,
                    max_iterations=self._max_iterations,
                    order=self._subset_order,
                    warp_0=p_0,
                    method=self._method,
                    tolerance=self._tolerance,
                )
                if self._subsets[idx].data["solved"]:  # Check against tolerance.
                    self._store_variables(idx=idx, flag=1)
                elif self._subsets[idx].data["unsolvable"]:
                    self._status = 5
                    self._unsolvable = True
                    return
                else:
                    # 2. Use projected pre-conditioning.
                    diff = self._nodes[idx] - self._nodes[cur_idx]
                    p = self._subsets[cur_idx].data["results"]["p"]
                    if np.shape(p)[0] == 6:
                        p_0[0] = p[0] + p[2] * diff[0] + p[3] * diff[1]
                        p_0[1] = p[1] + p[4] * diff[0] + p[5] * diff[1]
                    elif np.shape(p)[0] == 12:
                        p_0[0] = (
                            p[0]
                            + p[2] * diff[0]
                            + p[4] * diff[1]
                            + 0.5 * p[6] * diff[0] ** 2
                            + p[8] * diff[0] * diff[1]
                            + 0.5 * p[10] * diff[1] ** 2
                        )
                        p_0[1] = (
                            p[1]
                            + p[3] * diff[0]
                            + p[5] * diff[1]
                            + 0.5 * p[7] * diff[0] ** 2
                            + p[9] * diff[0] * diff[1]
                            + 0.5 * p[11] * diff[1] ** 2
                        )
                        p_0[2] = p[2] + p[6] * diff[0] + p[8] * diff[1]
                        p_0[3] = p[3] + p[7] * diff[0] + p[9] * diff[1]
                        p_0[4] = p[4] + p[8] * diff[1] + p[10] * diff[1]
                        p_0[5] = p[5] + p[9] * diff[1] + p[11] * diff[1]
                    self._subsets[idx].solve(
                        max_norm=self._max_norm,
                        max_iterations=self._max_iterations,
                        warp_0=p_0,
                        method=self._method,
                        tolerance=self._tolerance,
                        order=self._subset_order,
                    )
                    if self._subsets[idx].data["solved"]:  # Check against tolerance.
                        self._store_variables(idx=idx, flag=1)
                    elif self._subsets[idx].data["unsolvable"]:
                        self._status = 5
                        self._unsolvable = True
                        return
                    else:
                        # 3. Use the NCC initial guess.
                        self._subsets[idx].solve(
                            max_norm=self._max_norm,
                            max_iterations=self._max_iterations,
                            warp_0=np.zeros(np.shape(p_0)),
                            method=self._method,
                            tolerance=self._tolerance,
                            order=self._subset_order,
                        )
                        if self._subsets[idx].data["solved"]:
                            self._store_variables(idx=idx, flag=1)
                        elif self._subsets[idx].data["unsolvable"] or np.isnan(
                            self._subsets[idx].data["results"]["C_ZNCC"]
                        ):
                            self._status = 5
                            self._unsolvable = True
                            return
                        else:
                            self._C_ZNCC[idx] = self._subsets[idx].data["results"][
                                "C_ZNCC"
                            ]

    def _store_variables(self, idx, flag):
        """

        Private method to store variables.

        Parameters
        ----------
        idx : int
            Index of current subset.
        flag : int
            -1 : solved and stored
            0  : unsolved
            1  : solved and unstored

        """

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
