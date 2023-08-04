"""

Sequence module for geopyv.

"""
import logging
import numpy as np
import geopyv as gp
from geopyv.object import Object
import re
import glob
import traceback
from alive_progress import alive_bar

log = logging.getLogger(__name__)


class SequenceBase(Object):
    """

    Sequence base class to be used as a mixin. Contains plot functionality.

    """

    def __init__(self):
        super().__init__(object_type="Sequence")
        """

        Sequence base class initialiser.

        """

    def inspect(
        self,
        *,
        mesh_index=None,
        subset_index=None,
        show=True,
        block=True,
        save=None,
    ):
        """Method to show the sequence and associated mesh and subset quality metrics using
        :mod: `~geopyv.plots.inspect_subset` for subsets and
        :mod: `~geopyv.plots.inspect_mesh` for meshes.

        Parameters
        ----------
        mesh_index : int
            Index of the mesh to inspect.
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

        # Check if solved.
        if self.data["solved"] is not True:
            log.error(
                "Sequence not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.sequence.Sequence.solve()` to solve."
            )
            raise ValueError(
                "Sequence not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.sequence.Sequence.solve()` to solve."
            )

        # Check mesh_index input.
        self._report(
            gp.check._check_type(mesh_index, "mesh_index", [int, type(None)]),
            "TypeError",
        )
        if mesh_index:
            self._report(
                gp.check._check_index(mesh_index, "mesh_index", 0, self.data["meshes"]),
                "IndexError",
            )

        # Load/access selected mesh object.
        mesh_obj = self._load_mesh(mesh_index)

        # Check remaining input.
        self._report(
            gp.check._check_type(subset_index, "subset_index", [int, type(None)]),
            "TypeError",
        )
        if subset_index:
            self._report(
                gp.check._check_index(
                    subset_index, "subset_index", 0, mesh_obj["results"]["subsets"]
                ),
                "IndexError",
            )
        self._report(gp.check._check_type(show, "show", [bool]), "TypeError")
        self._report(gp.check._check_type(block, "block", [bool]), "TypeError")
        self._report(gp.check._check_type(save, "save", [str, type(None)]), "TypeError")

        # Inspect.
        if subset_index is not None:
            subset_data = mesh_obj["results"]["subsets"][subset_index]
            mask = np.asarray(mesh_obj["mask"])
            log.info(
                "Inspecting subset {subset} of mesh {mesh}...".format(
                    subset=subset_index, mesh=mesh_index
                )
            )
            fig, ax = gp.plots.inspect_subset(
                data=subset_data,
                mask=mask,
                show=show,
                block=block,
                save=save,
            )
            return fig, ax
        else:
            log.info("Inspecting mesh {mesh}...".format(mesh=mesh_index))
            fig, ax = gp.plots.inspect_mesh(
                data=mesh_obj,
                show=show,
                block=block,
                save=save,
            )
            return fig, ax

    def convergence(
        self,
        *,
        mesh_index=None,
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
        mesh_index : int
            Index of the mesh to inspect.
        subset_index : int, optional
            Index of the subset to inspect. If `None',
            the convergence plot is for the mesh instead.
        quantity : str, optional
            Selector for histogram convergence property if
            the convergence plot is for mesh. Defaults
            to `C_ZNCC` if left as default None.
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
                "Sequence not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.sequence.Sequence.solve()` to solve."
            )
            raise ValueError(
                "Sequence not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.sequence.Sequence.solve()` to solve."
            )

        # Check mesh_index input.
        self._report(
            gp.check._check_type(mesh_index, "mesh_index", [int, type(None)]),
            "TypeError",
        )
        if mesh_index:
            self._report(
                gp.check._check_index(mesh_index, "mesh_index", 0, self.data["meshes"]),
                "IndexError",
            )

        # Load/access selected mesh object.
        mesh_obj = self._load_mesh(mesh_index)

        # Check remaining input.
        self._report(
            gp.check._check_type(subset_index, "subset_index", [int, type(None)]),
            "TypeError",
        )
        if subset_index:
            self._report(
                gp.check._check_index(
                    subset_index, "subset_index", 0, mesh_obj["results"]["subsets"]
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
                (
                    "Generating convergence plots for subset {subset} "
                    "of mesh {mesh}..."
                ).format(subset=subset_index, mesh=mesh_index)
            )
            fig, ax = gp.plots.convergence_subset(
                mesh_obj["results"]["subsets"][subset_index],
                show=show,
                block=block,
                save=save,
            )
            return fig, ax
        else:
            if quantity is None:
                quantity = "C_ZNCC"
            elif quantity not in ["C_ZNCC", "iterations", "norm"]:
                log.warning(
                    (
                        "Invalid quantity specified: {quantity}. "
                        "Reverting to C_ZNCC."
                    ).format(quantity=quantity)
                )
                quantity = "C_ZNCC"
            log.info(
                (
                    "Generating {quantity} convergence histogram " "for mesh {mesh}..."
                ).format(quantity=quantity, mesh=mesh_index)
            )
            fig, ax = gp.plots.convergence_mesh(
                data=mesh_obj,
                quantity=quantity,
                show=show,
                block=block,
                save=save,
            )
            return fig, ax

    def contour(
        self,
        *,
        mesh_index=None,
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
        mesh_index : int
            Index of the mesh to inspect.
        quantity : str, optional
            Selector for contour parameter. Must be in:
            [`C_ZNCC`, `u`, `v`, `u_x`, `v_x`,`u_y`,`v_y`, `R`]
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
            * Can only be used once the sequence has been solved using the
              :meth:`~geopyv.sequence.Sequence.solve` method.

        .. seealso::
            :meth:`~geopyv.plots.contour_sequence`

        """
        # Check if solved.
        if self.data["solved"] is not True:
            log.error(
                "Sequence not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.sequence.Sequence.solve()` to solve."
            )
            raise ValueError(
                "Sequence not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.sequence.Sequence.solve()` to solve."
            )

        # Check input.
        self._report(
            gp.check._check_type(mesh_index, "mesh_index", [int, type(None)]),
            "TypeError",
        )
        if mesh_index:
            self._report(
                gp.check._check_index(mesh_index, "mesh_index", 0, self.data["meshes"]),
                "IndexError",
            )

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

        # Load/access selected mesh object.
        mesh_obj = self._load_mesh(mesh_index)

        log.info(
            "Generating {quantity} contour plot for mesh {mesh}...".format(
                quantity=quantity, mesh=mesh_index
            )
        )
        fig, ax = gp.plots.contour_mesh(
            data=mesh_obj,
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
        mesh_index=None,
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
        mesh_index : int
            Index of the mesh to inspect.
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
                "Sequence not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.sequence.Sequence.solve()` to solve."
            )
            raise ValueError(
                "Sequence not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.sequence.Sequence.solve()` to solve."
            )

        # Check input.
        self._report(
            gp.check._check_type(mesh_index, "mesh_index", [int, type(None)]),
            "TypeError",
        )
        if mesh_index:
            self._report(
                gp.check._check_index(mesh_index, "mesh_index", 0, self.data["meshes"]),
                "IndexError",
            )
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

        # Load/access selected mesh object.
        mesh_obj = self._load_mesh(mesh_index)

        # Plot quiver.
        log.info("Generating quiver plot for mesh {mesh}...".format(mesh=mesh_index))
        fig, ax = gp.plots.quiver_mesh(
            data=mesh_obj,
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

    def _load_mesh(self, mesh_index, verbose=True):
        if self.data["file_settings"]["save_by_reference"]:
            return gp.io.load(
                filename=self.data["file_settings"]["mesh_dir"]
                + self.data["meshes"][mesh_index],
                verbose=verbose,
            ).data
        else:
            return self.data["meshes"][mesh_index]

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


class Sequence(SequenceBase):
    def __init__(
        self,
        *,
        image_dir=".",
        common_name="",
        file_format=".jpg",
        target_nodes=1000,
        boundary_obj=None,
        exclusion_objs=[],
        size_lower_bound=1.0,
        size_upper_bound=1000.0,
        save_by_reference=False,
        mesh_dir=".",
    ):
        """Initialisation of geopyv sequence object.

        Parameters
        ----------
        image_dir : str, optional
            Directory of images. Defaults to current working directory.
        file_format : str, optional
            Image file type. Options are ".jpg", ".png" or ".bmp". Defaults to .jpg.
        target_nodes : int
            Target node number. Defaults to 1000.
        boundary : `numpy.ndarray` (N,2)
            Array of coordinates to define the mesh boundary.
            Must be specified in clockwise or anti-clockwise order.
        exclusions: list, optional
            List of `numpy.ndarray` to define the mesh exclusions.
        size_lower_bound : int, optional
            Lower bound on element size. Defaults to a value of 1.
        upper_lower_bound : int, optional
            Lower bound on element size. Defaults to a value of 1000.
        save_by_reference : bool, optional
            False - sequence object stores meshes.
            True - sequence object stores mesh references.

        Attributes
        ----------
        data : dict
            Data object containing all settings and results.
            See the data structure :ref:`here <sequence_data_structure>`.
        solved : bool
            Boolean to indicate if the sequence has been solved.
        """

        # Set initialised boolean.
        self.initialised = False

        # Check types.
        self._report(gp.check._check_type(image_dir, "image_dir", [str]), "TypeError")
        if self._report(gp.check._check_path(image_dir, "image_dir"), "Warning"):
            image_dir = gp.io._get_image_dir()
        image_dir = gp.check._check_character(image_dir, "/", -1)
        self._report(
            gp.check._check_type(file_format, "file_format", [str]), "TypeError"
        )
        file_format = gp.check._check_character(file_format, ".", 0)
        self._report(
            gp.check._check_value(
                file_format,
                "file_format",
                [".jpg", ".png", ".bmp", ".JPG", ".PNG", ".BMP"],
            ),
            "ValueError",
        )
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
        self._report(
            gp.check._check_type(save_by_reference, "save_by_reference", [bool]),
            "TypeError",
        )
        self._report(gp.check._check_type(mesh_dir, "mesh_dir", [str]), "TypeError")
        if self._report(gp.check._check_path(mesh_dir, "mesh_dir"), "Warning"):
            mesh_dir = gp.io._get_mesh_dir()
        mesh_dir = gp.check._check_character(mesh_dir, "/", -1)

        # Store variables.
        self._image_dir = image_dir
        self._file_format = file_format
        self._common_name = common_name
        try:
            _images = glob.glob(
                self._common_name + "*" + self._file_format, root_dir=self._image_dir
            )
            if np.shape(_images)[0] < 2:
                log.error("Insufficient number of images in specified folder.")
                raise FileExistsError(
                    "Insufficient number of images in specified folder."
                )
            _image_indices_unordered = [int(re.findall(r"\d+", x)[-1]) for x in _images]
            _image_indices_arguments = np.argsort(_image_indices_unordered)
            self._images = [_images[index] for index in _image_indices_arguments]
            self._image_indices = np.sort(_image_indices_unordered)

        except Exception:
            log.error(
                (
                    "No files found under: \n{address}. "
                    "Please check file names and directories."
                ).format(
                    address=self._image_dir
                    + self._common_name
                    + "*"
                    + self._file_format
                )
            )
            print(traceback.format_exc())
            return
        self._target_nodes = target_nodes
        self._boundary_obj = boundary_obj
        self._exclusion_objs = exclusion_objs
        self._size_lower_bound = size_lower_bound
        self._size_upper_bound = size_upper_bound
        self._save_by_reference = save_by_reference
        self._mesh_dir = mesh_dir
        if self._save_by_reference:
            meshes = []
        else:
            meshes = np.empty(np.shape(self._image_indices)[0] - 1, dtype=dict)
        self.solved = False
        self._unsolvable = False

        # Data.
        file_settings = {
            "image_dir": self._image_dir,
            "images": self._images,
            "file_format": self._file_format,
            "image_indices": self._image_indices,
            "save_by_reference": self._save_by_reference,
            "mesh_dir": self._mesh_dir,
        }
        mesh_settings = {
            "target_nodes": self._target_nodes,
            "boundary_obj": self._boundary_obj,
            "exclusion_objs": self._exclusion_objs,
            "size_lower_bound": self._size_lower_bound,
            "size_upper_bound": self._size_upper_bound,
        }
        self.data = {
            "type": "Sequence",
            "solved": self.solved,
            "unsolvable": self._unsolvable,
            "meshes": meshes,
            "mesh_settings": mesh_settings,
            "file_settings": file_settings,
        }

        self._initialised = True

    def solve(
        self,
        *,
        seed_coord=None,
        seed_warp=None,
        template=None,
        max_norm=1e-5,
        max_iterations=50,
        adaptive_iterations=0,
        correction=True,
        method="ICGN",
        mesh_order=2,
        subset_order=1,
        tolerance=0.75,
        seed_tolerance=0.9,
        alpha=0.5,
        guide=False,
        sequential=False,
    ):
        """
        Method to solve for the sequence.

        Parameters
        ----------
        seed_coord : numpy.ndarray (2,)
            Seed coordinate for reliability-guided mesh solving.
        template : gp.template.Template object, optional
            subset template. Defaults to Circle(50).
        max_norm : float, optional
            Exit criterion for norm of increment in warp function.
            Defaults to value of
            :math:`1 cdot 10^{-3}`.
        max_iterations : int, optional
            Exit criterion for number of Gauss-Newton iterations.
            Defaults to value of 15.
        mesh_order : int, optional
            Mesh element order. Options are 1 and 2. Defaults to 2.
        subset_order : int, optional
            Warp function order. Options are 1 and 2. Defaults to 1.
        tolerance: float, optional
            Correlation coefficient tolerance. Defaults to a value of 0.7.
        seed_tolerance: float, optional
            Correlation coefficient tolerance for the seed subset.
            Defaults to a value of 0.9.
        method : str
            Solution method. Options are FAGN and ICGN.
            Default is ICGN since it is faster.
        adaptive_iterations : int, optional
            Number of mesh adaptivity iterations to perform.
            Defaults to a value of 0.
        alpha : float, optional
            Mesh adaptivity control parameter.
            Defaults to a value of 0.5.

        Returns
        -------

        solved : bool
            Boolean to indicate if the subset instance has been solved.
        """
        # Check if solved.
        if self.data["solved"] is True:
            log.error("Sequence has already been solved. Cannot be solved again.")
            return

        # Check inputs.
        if self._report(
            gp.check._check_type(seed_coord, "seed_coord", [np.ndarray]), "Warning"
        ):
            seed_coord = gp.gui.selectors.coordinate.CoordinateSelector()
        elif self._report(gp.check._check_dim(seed_coord, "seed_coord", 1), "Warning"):
            seed_coord = gp.gui.selectors.coordinate.CoordinateSelector()
        elif self._report(
            gp.check._check_axis(seed_coord, "seed_coord", 0, [2]), "Warning"
        ):
            seed_coord = gp.gui.selectors.coordinate.CoordinateSelector()
        elif gp.check._check_dtype(seed_coord, "seed_coord", [float]):
            try:
                np.float64(seed_coord)
                self._report(
                    gp.check._dconversion(seed_coord, "seed_coord", np.float64),
                    "Warning",
                )
            except Exception:
                seed_coord(gp.gui.selectors.coordinate.CoordinateSelector())
        if template is None:
            template = gp.templates.Circle(50)
        types = [gp.templates.Circle, gp.templates.Square]
        self._report(gp.check._check_type(template, "template", types), "TypeError")
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
        check = gp.check._check_type(alpha, "alpha", [float])
        if check:
            try:
                alpha = float(alpha)
                self._report(gp.check._conversion(alpha, "alpha", float), "Warning")
            except Exception:
                self._report(check, "TypeError")
        self._report(gp.check._check_range(alpha, "alpha", 0.0, 1.0), "ValueError")
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
            seed_warp = np.zeros(12)

        # Store variables.
        self._seed_coord = seed_coord
        self._template = template
        self._max_iterations = max_iterations
        self._max_norm = max_norm
        self._adaptive_iterations = adaptive_iterations
        self._correction = correction
        self._method = method
        self._subset_order = subset_order
        self._mesh_order = mesh_order
        self._tolerance = tolerance
        self._seed_tolerance = seed_tolerance
        self._alpha = alpha
        self._guide = guide
        self._sequential = sequential
        self._seed_warp = np.zeros(6 * self._subset_order)

        # Solve.
        _f_index = 0
        _g_index = 1
        _seed_coord_t = self._seed_coord
        _seed_displacement = np.zeros(2)
        _f_img = gp.image.Image(self._image_dir + self._images[_f_index])
        _g_img = gp.image.Image(self._image_dir + self._images[_g_index])
        while _g_index < len(self._image_indices):
            log.info(
                "Solving for image pair {}-{}.".format(
                    self._image_indices[_f_index],
                    self._image_indices[_g_index],
                )
            )
            mesh = gp.mesh.Mesh(
                f_img=_f_img,
                g_img=_g_img,
                target_nodes=self._target_nodes,
                boundary_obj=self._boundary_obj,
                exclusion_objs=self._exclusion_objs,
                size_lower_bound=self._size_lower_bound,
                size_upper_bound=self._size_upper_bound,
                mesh_order=self._mesh_order,
            )  # Initialise mesh object.
            mesh.solve(
                seed_coord=self._seed_coord,
                seed_warp=self._seed_warp,
                template=self._template,
                max_iterations=self._max_iterations,
                max_norm=self._max_norm,
                adaptive_iterations=self._adaptive_iterations,
                correction=self._correction,
                method=self._method,
                subset_order=self._subset_order,
                tolerance=self._tolerance,
                seed_tolerance=self._seed_tolerance,
                alpha=self._alpha,
            )  # Solve mesh.
            if mesh.solved:
                # Save/store mesh.
                if self._save_by_reference:  # Save...
                    gp.io.save(
                        object=mesh,
                        filename=self._mesh_dir
                        + "mesh_"
                        + str(self._image_indices[_f_index])
                        + "_"
                        + str(self._image_indices[_g_index]),
                    )
                    self.data["meshes"].append(
                        "mesh_"
                        + str(self._image_indices[_f_index])
                        + "_"
                        + str(self._image_indices[_g_index])
                    )
                else:  # ...or store.
                    self.data["meshes"][_g_index - 1] = mesh.data
                # Iterate target index.
                _g_index += 1  # Iterate the target image index.
                # Check for end.
                if _g_index != len(self._image_indices):
                    del _g_img
                    _g_img = gp.image.Image(self._image_dir + self._images[_g_index])
                else:
                    self.solved = True
                    break
                # Setup for next mesh.
                # Deformation preconditioning (if guided).
                if self._guide:
                    seed = gp.particle.Particle(series=mesh, coordinate_0=_seed_coord_t)
                    seed.solve()
                    _seed_displacement = seed.data["results"]["warps"][1, :2]
                    self._seed_warp[
                        : 6 * min(self._mesh_order, self._subset_order)
                    ] = seed.data["results"]["warps"][
                        1, : 6 * min(self._mesh_order, self._subset_order)
                    ]
                # Reference updating (if sequential).
                if self._sequential:
                    _f_index = _g_index - 1
                    del _f_img
                    _f_img = gp.image.Image(self._image_dir + self._images[_f_index])
                    _seed_coord_t += _seed_displacement
                    _seed_displacement = np.zeros(2)
            elif _f_index + 1 < _g_index:
                _f_index = _g_index - 1
                del _f_img
                _f_img = gp.image.Image(self._image_dir + self._images[_f_index])
                _seed_coord_t += _seed_displacement
                _seed_displacement = np.zeros(2)
            else:
                log.error(
                    "Mesh for consecutive image pair {a}-{b} is unsolvable. "
                    "Sequence curtailed.".format(
                        a=self._image_indices[_f_index],
                        b=self._image_indices[_g_index],
                    )
                )
                self.data["meshes"] = np.asarray(self.data["meshes"])
                self._unsolvable = True
                del mesh
                return self.solved
            del mesh
        del _f_img

        # Pack data.
        self.data["solved"] = self.solved
        self.data["unsolvable"] = self._unsolvable
        self.data["meshes"] = np.asarray(self.data["meshes"])
        return self.solved

    def load(self):
        try:
            _meshes = glob.glob("*.pyv", root_dir=self._mesh_dir)
        except Exception:
            msg = ("Cannot find pyv files with specified path: {}").format(
                self._mesh_dir + "*.pyv"
            )
            log.error(msg)
            raise FileExistsError(msg)
        try:
            _mesh_tars = [int(re.findall(r"\d+", x)[-1]) for x in _meshes]
        except Exception:
            msg = (
                "Cannot extract integers from file names. "
                "Please ensure file names are of the form: `mesh_X_Y.pyv`, "
                "where X and Y are integers."
            )
            log.error(msg)
            raise ValueError(msg)
        _mesh_indices_arguments = np.argsort(_mesh_tars)
        self._meshes = np.asarray(
            [_meshes[index] for index in _mesh_indices_arguments], dtype=object
        )

        self._images = self._images[: np.shape(self._meshes)[0] + 1]
        self.data["solved"] = True
        self.data["unsolvable"] = False
        self.data["meshes"] = self._meshes
        self.data["file_settings"]["images"]


class SequenceResults(SequenceBase):
    """
    SequenceResults class for geopyv.

    Parameters
    ----------
    data : dict
        geopyv data dict from Sequence object.


    Attributes
    ----------
    data : dict
        geopyv data dict from Sequence object.

    """

    def __init__(self, data):
        """Initialisation of geopyv SequenceResults class."""
        self.data = data

    def load(self, ref=True):
        """Load all meshes in sequence object mesh directory."""
        try:
            _meshes = glob.glob(
                "*.pyv", root_dir=self.data["file_settings"]["mesh_dir"]
            )
            _mesh_tars = [int(re.findall(r"\d+", x)[-1]) for x in _meshes]
            _mesh_indices_arguments = np.argsort(_mesh_tars)
            self._meshes = np.asarray(
                [_meshes[index] for index in _mesh_indices_arguments], dtype=object
            )
        except Exception:
            log.error(
                "Issues encountered recognising mesh file names. "
                "Please refer to the documentation for naming guidance."
            )
        self._images = self.data["file_settings"]["images"][
            : np.shape(self._meshes)[0] + 1
        ]
        self.data["solved"] = True
        self.data["unsolvable"] = False
        self.data["file_settings"]["images"] = self._images
        if ref is not True:
            with alive_bar(
                np.shape(self._meshes)[0],
                dual_line=True,
                bar="blocks",
                title="Loading meshes...",
            ) as bar:
                for i in range(np.shape(self._meshes)[0]):
                    self._meshes[i] = self._load_mesh(mesh_index=i, verbose=False)
                    bar()
            self.data["file_settings"]["save_by_reference"] = False
        self.data["meshes"] = self._meshes
