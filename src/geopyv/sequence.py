"""

Sequence module for geopyv.

"""
import logging
import numpy as np
import geopyv as gp
from geopyv.object import Object
import re
import os
import sys
import glob
import matplotlib.pyplot as plt

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

    def inspect(self, mesh=None, subset=None, show=True, block=True, save=None):
        """Method to show the sequence and associated mesh and subset properties."""

        # If a mesh index is given, inspect the mesh.
        if mesh is not None:
            if mesh >= 0 and mesh < len(self.data["meshes"]):
                # If a subset index is given, inspect the subset of the mesh.
                if subset is not None:
                    if subset >= 0 and subset < len(
                        self.data["meshes"][mesh]["results"]["subsets"]
                    ):
                        subset_data = self.data["meshes"][mesh]["results"]["subsets"][
                            subset
                        ]
                        mask = np.asarray(self.data["meshes"][mesh]["mask"])

                        fig, ax = gp.plots.inspect_subset(
                            data=subset_data,
                            mask=mask,
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
                    fig, ax = gp.plots.inspect_mesh(
                        data=self.data["meshes"][mesh],
                        show=show,
                        block=block,
                        save=save,
                    )
                    return fig, ax
            else:
                log.error(
                    "Mesh index is out of the range of the sequence object contents."
                )
        # Otherwise inspect the sequence.
        else:
            log.error("No mesh or subset index provided.")
            # fig, ax = gp.plots.inspect_sequence(
            #    data=self.data, show=show, block=block, save=save
            # )
            # return fig, ax

    def convergence(
        self, mesh=None, subset=None, quantity=None, show=True, block=True, save=None
    ):
        """
        Method to plot the rate of convergence for a mesh or subset.
        """
        # If a mesh index is given, inspect the mesh.
        if mesh is not None:
            if mesh >= 0 and mesh < len(self.data["meshes"]):
                if subset is not None:
                    if subset >= 0 and subset < len(
                        self.data["meshes"][mesh]["results"]["subsets"]
                    ):
                        fig, ax = gp.plots.convergence_subset(
                            self.data["meshes"][mesh]["results"]["subsets"][subset],
                            show=show,
                            block=block,
                            save=save,
                        )
                        return fig, ax
                    else:
                        log.error(
                            "Subset index is out of the range of the mesh object contents."
                        )
                # Otherwise inspect mesh.
                else:
                    if quantity is None:
                        quantity = "C_ZNCC"
                    fig, ax = gp.plots.convergence_mesh(
                        data=self.data["meshes"][mesh],
                        quantity=quantity,
                        show=show,
                        block=block,
                        save=save,
                    )
                    return fig, ax
            else:
                log.error(
                    "Mesh index is out of the range of the sequence object contents."
                )
        else:
            log.error("No mesh or subset index provided.")

    def contour(
        self,
        mesh_index=None,
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
        if mesh_index is not None:
            if mesh_index >= 0 and mesh_index < len(self.data["meshes"]):
                if quantity is not None:
                    fig, ax = gp.plots.contour_mesh(
                        data=self.data["meshes"][mesh_index],
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
            else:
                log.error(
                    "Mesh index is out of the range of the sequence object contents."
                )
        else:
            log.error("No mesh index provided.")

    def quiver(
        self,
        mesh_index=None,
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
        if mesh_index is not None:
            if mesh_index >= 0 and mesh_index < len(self.data["meshes"]):
                fig, ax = gp.plots.quiver_mesh(
                    data=self.data["meshes"][mesh_index],
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
            else:
                log.error(
                    "Mesh index is out of the range of the sequence object contents."
                )
        else:
            log.error("No mesh index provided.")


class Sequence(SequenceBase):
    def __init__(
        self,
        *,
        image_folder=".",
        common_name="",
        image_file_type=".jpg",
        target_nodes=1000,
        boundary=None,
        exclusions=[],
        size_lower_bound=1,
        size_upper_bound=1000,
    ):
        """Initialisation of geopyv sequence object.

        Parameters
        ----------
        image_folder : str, optional
            Directory of images. Defaults to current working directory.
        image_file_type : str, optional
            Image file type. Options are ".jpg", ".png" or ".bmp". Defaults to .jpg.
        target_nodes : int
            Target node number. Defaults to 1000.
        boundary : `numpy.ndarray` (N,2)
            Array of coordinates to define the mesh boundary. Must be specified in clockwise or anti-clockwise order.
        exclusions: list, optional
            List of `numpy.ndarray` to define the mesh exclusions.
        size_lower_bound : int, optional
            Lower bound on element size. Defaults to a value of 1.
        upper_lower_bound : int, optional
            Lower bound on element size. Defaults to a value of 1000.

        Attributes
        ----------
        data : dict
            Data object containing all settings and results.
            See the data structure :ref:`here <sequence_data_structure>`.
        solved : bool
            Boolean to indicate if the sequence has been solved.
        """

        self.initialised = False
        # Check types.
        if type(image_folder) != str:
            log.error("image_folder type invalid. Expected a string.")
            return
        elif os.path.isdir(image_folder) is False:
            log.error("image_folder does not exist.")
            return
        if type(image_file_type) != str:
            log.error("image_file_type type invalid. Expected a string.")
            return
        elif image_file_type not in [".jpg", ".png", ".bmp"]:
            log.error("image_file_type invalid. Expected: '.jpg', '.png', or '.bmp'.")
            return
        if type(target_nodes) != int:
            try:
                target_nodes = int(target_nodes)
            except:
                log.error("Target nodes type invalid. Expected an integer.")
                return
        if type(boundary) != np.ndarray:
            log.error(
                "Boundary coordinate array type invalid. Expected a numpy.ndarray."
            )
            return
        if np.shape(boundary)[1] != 2:
            log.error(
                "Boundary coordinate array shape invalid.Must be numpy.ndarray of size (n, 2)."
            )
            return
        if type(exclusions) != list:
            log.error("Exclusions type invalid. Expected a list.")
            return
        for exclusion in exclusions:
            if type(exclusion) != np.ndarray:
                log.error(
                    "Exclusion coordinate array type invalid. Expected a numpy.ndarray."
                )
                return
            if np.ndim(exclusion) != 2:
                log.error(
                    "Exclusion array dimensions invalid. Expected a 2D numpy.ndarray."
                )
                return
            if np.shape(exclusion)[1] != 2:
                log.error(
                    "Exclusion coordinate array shape invalid. Must be numpy.ndarray of size (n, 2)."
                )
                return

        if isinstance(size_lower_bound, (int, float)) is False:
            log.error("size_lower_bound type invalid. Expected an integer or float.")
            return
        if isinstance(size_upper_bound, (int, float)) is False:
            log.error("size_lower_bound type invalid. Expected an integer or float.")
            return

        # Store variables.
        self._image_folder = image_folder
        self._image_file_type = image_file_type
        self._common_name = common_name
        platform = sys.platform
        if platform == "linux" or platform == "linux2" or platform == "darwin":
            split = "/"
        elif platform == "win32":
            split = r"\\"
        try:
            _images = glob.glob(
                self._image_folder
                + split
                + self._common_name
                + "*"
                + self._image_file_type
            )
            _image_indices_unordered = np.argsort(
                [int(re.findall(r"\d+", x)[-1]) for x in _images]
            )
            self._images = [_images[index] for index in _image_indices_unordered]
            self._image_indices = np.sort(_image_indices_unordered)
        except:
            log.error(
                "Issues encountered recognising image file names. Please refer to the documentation for naming guidance."
            )
            return
        self._number_images = np.shape(self._image_indices)[0]
        self._target_nodes = target_nodes
        self._boundary = boundary
        self._exclusions = exclusions
        self._size_lower_bound = size_lower_bound
        self._size_upper_bound = size_upper_bound
        self.solved = False
        self._unsolvable = False

        # Data.
        self.data = {
            "type": "Sequence",
            "solved": self.solved,
            "unsolvable": self._unsolvable,
            "meshes": np.empty(self._number_images - 1, dtype=dict),
            "image_folder": self._image_folder,
            "images": self._images,
            "image_indices": self._image_indices,
            "number_images": self._number_images,
            "image_file_type": self._image_file_type,
            "target_nodes": self._target_nodes,
            "boundary": self._boundary,
            "exclusions": self._exclusions,
            "size_lower_bound": self._size_lower_bound,
            "size_upper_bound": self._size_upper_bound,
        }

        self._initialised = True

    def solve(
        self,
        *,
        seed_coord=None,
        template=None,
        max_norm=1e-3,
        max_iterations=15,
        adaptive_iterations=0,
        method="ICGN",
        mesh_order=2,
        subset_order=1,
        tolerance=0.7,
        alpha=0.5,
        track="fixed",
        mesh_save=False,
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
            Exit criterion for norm of increment in warp function. Defaults to value of
            :math:`1 \cdot 10^{-3}`.
        max_iterations : int, optional
            Exit criterion for number of Gauss-Newton iterations. Defaults to value
            of 15.
        mesh_order : int, optional
            Mesh element order. Options are 1 and 2. Defaults to 2.
        subset_order : int, optional
            Warp function order. Options are 1 and 2. Defaults to 1.
        tolerance: float, optional
            Correlation coefficient tolerance. Defaults to a value of 0.7.
        method : str
            Solution method. Options are FAGN and ICGN.
            Default is ICGN since it is faster.
        adaptive_iterations : int, optional
            Number of mesh adaptivity iterations to perform. Defaults to a value of 0.
        alpha : float, optional
            Mesh adaptivity control parameter. Defaults to a value of 0.5.
        track : str, optional
            Mesh boundary tracking at reference image updates. Options are:
            "fixed" - no movement,
            "move" - movement of initially defined boundary points, or
            "add" - inclusion of generated boundary points.
        mesh_save: bool, optional
            Option to save individual mesh .pyv files. Defaults to False.

        Returns
        -------
        solved : bool
            Boolean to indicate if the subset instance has been solved.
        """

        # Check inputs.
        if type(seed_coord) != np.ndarray:
            try:
                seed_coord = np.asarray(seed_coord)
            except:
                log.error("Seed coordinate type invalid. Expected a numpy.ndarray. ")
                return False
        if np.ndim(seed_coord) != 1:
            log.error(
                "Seed coordinate array dimensions invalid. Expected a 1D numpy.ndarray."
            )
        if type(adaptive_iterations) != int:
            try:
                adaptive_iterations = int(adaptive_iterations)
            except:
                log.error("Adaptive iterations type invalid. Expected an integer.")
                return False
        if adaptive_iterations < 0:
            log.error(
                "Adaptive iterations out of range. Expected an integer greater than or equal to zero."
            )
            return False
        if template is None:
            template = gp.templates.Circle(50)
        elif (
            type(template) != gp.templates.Circle
            and type(template) != gp.templates.Square
        ):
            log.error("Template is not a type defined in geopyv.templates.")
            return False
        if type(max_iterations) != int:
            try:
                max_iterations = int(max_iterations)
            except:
                log.error("Maximum iterations type invalid. Expected an integer.")
                return False
        if max_iterations < 1:
            log.error("Maximum iterations out of range. Expected an integer >=1.")
            return False
        if type(max_norm) != float:
            try:
                max_norm = float(max_norm)
            except:
                log.error("Maximum norm type invalid. Expected a float.")
                return False
        if max_norm < 0:
            log.error("Maximum norm out of range. Expected a float >=0.0.")
            return False
        if method not in ["FAGN", "ICGN"]:
            log.error("Method invalid. Expected 'FAGN' or 'ICGN'.")
            return False
        if subset_order != 1 and subset_order != 2:
            log.error("Subset order invalid. Must be 1 or 2.")
            return False
        if mesh_order != 1 and mesh_order != 2:
            log.error("Mesh order invalid. Must be 1 or 2.")
            return False
        if type(tolerance) != float:
            try:
                tolerance = float(tolerance)
            except:
                log.error("Tolerance type invalid. Expected a float.")
                return False
        if tolerance < 0 or tolerance > 1:
            log.error("Tolerance out of range. Expected a float 0.0-1.0")
            return False
        if type(alpha) != float:
            try:
                alpha = float(alpha)
            except:
                log.error("Alpha type invalid. Expected a float.")
                return False
        if alpha > 1 or alpha < 0:
            log.error("Alpha out of range. Expected a float 0.0-1.0.")
            return False
        if track not in ["fixed", "move", "add"]:
            log.error("Track invalid. Must be 'fixed', 'move' or 'add'.")
        if type(mesh_save) != bool:
            log.error("Mesh save type invalid. Must be a boolean. ")

        # Store variables.
        self._seed_coord = seed_coord
        self._template = template
        self._max_iterations = max_iterations
        self._max_norm = max_norm
        self._adaptive_iterations = adaptive_iterations
        self._method = method
        self._subset_order = subset_order
        self._mesh_order = mesh_order
        self._tolerance = tolerance
        self._alpha = alpha
        self._track = track
        self._p_0 = np.zeros(6 * self._subset_order)

        # Solve.
        _f_index = 0
        _g_index = 1
        _f_img = gp.image.Image(self._images[_f_index])
        _g_img = gp.image.Image(self._images[_g_index])
        while _g_index < len(self._image_indices) - 1:
            log.info(
                "Solving for image pair {}-{}.".format(
                    self._image_indices[_f_index], self._image_indices[_g_index]
                )
            )
            mesh = gp.mesh.Mesh(
                f_img=_f_img,
                g_img=_g_img,
                target_nodes=self._target_nodes,
                boundary=self._boundary,
                exclusions=self._exclusions,
                size_lower_bound=self._size_lower_bound,
                size_upper_bound=self._size_upper_bound,
                mesh_order=self._mesh_order,
            )  # Initialise mesh object.
            mesh.solve(
                seed_coord=self._seed_coord,
                template=self._template,
                max_iterations=self._max_iterations,
                max_norm=self._max_norm,
                adaptive_iterations=self._adaptive_iterations,
                method=self._method,
                subset_order=self._subset_order,
                tolerance=self._tolerance,
                alpha=self._alpha,
            )  # Solve mesh.
            if mesh.solved:
                self.data["meshes"][_g_index - 1] = mesh.data
                if mesh_save:
                    gp.io.save(
                        object=mesh,
                        filename="mesh_"
                        + str(self._image_indices[_f_index])
                        + "_"
                        + str(self._image_indices[_g_index]),
                    )
                if track == "add":
                    self._boundary_tags = mesh._boundary_add_tags
                    self._exclusions_tags = mesh._exclusions_add_tags
                elif track == "move":
                    self._boundary_tags = mesh._boundary_move_tags
                    self._exclusions_tags = mesh._exclusions_move_tags
                _g_index += 1  # Iterate the target image index.
                del _g_img
                if _g_index != len(self._image_indices - 1):
                    _g_img = gp.image.Image(self._images[_g_index])
                else:
                    self.solved = True
            elif _f_index + 1 < _g_index:
                _f_index = _g_index - 1
                if self._track != "fixed":
                    self._tracking(_f_index)
                del _f_img
                _f_img = gp.image.Image(self._images[_f_index])
            else:
                log.error(
                    "Mesh for consecutive image pair {a}-{b} is unsolvable. Sequence curtailed.".format(
                        a=self._image_indices[_f_index], b=self._image_indices[_g_index]
                    )
                )
                self._unsolvable = True
                del mesh
                return self.solved
            del mesh
        del _f_img

        # Pack data.
        self.data["solved"] = self.solved
        self.data["unsolvable"] = self._unsolvable
        return self.solved

    def _tracking(self, _f_index):
        """
        Private method for tracking the movement of the mesh boundary and exclusions upon reference image updates.
        """

        log.info("Tracing boundary and exclusion displacements.")

        self._boundary = (
            self.data["meshes"][_f_index - 1]["nodes"][self._boundary_tags]
            + self.data["meshes"][_f_index - 1]["results"]["displacements"][
                self._boundary_tags
            ]
        )
        _exclusions = []
        for i in range(np.shape(self._exclusions)[0]):
            _exclusions.append(
                self.data["meshes"][_f_index - 1]["nodes"][self._exclusions_tags[i]]
                + self.data["meshes"][_f_index - 1]["results"]["displacements"][
                    self._exclusions_tags[i]
                ]
            )
        self._exclusions = _exclusions


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
