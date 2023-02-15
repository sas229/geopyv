"""

Sequence module for geopyv.

"""
import logging
import numpy as np
import geopyv as gp
from geopyv.object import Object
import re
import os

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
                print(mesh)
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
        image_file_type=".jpg",
        target_nodes=1000,
        boundary=None,
        exclusions=[],
        size_lower_bound=1,
        size_upper_bound=1000,
    ):
        """Initialisation of geopyv sequence object."""
        self.initialised = False
        # Check types.
        if type(image_folder) != str:
            log.error("image_folder type not recognised. " "Expected a string.")
            # return False
        elif os.path.isdir(image_folder) is False:
            log.error("image_folder does not exist.")
            # return False
        if type(image_file_type) != str:
            log.error("image_file_type type not recognised. " "Expected a string.")
            # return False
        elif image_file_type not in [".jpg", ".png", ".bmp"]:
            log.error(
                "image_file_type not recognised. "
                "Expected: '.jpg', '.png', or '.bmp'."
            )
            # return False
        if type(target_nodes) != int:
            log.error("Target nodes not of integer type.")
            # return False
        if type(boundary) != np.ndarray:
            log.error(
                "Boundary coordinate array of invalid type. " "Cannot initialise mesh."
            )
        if np.shape(boundary)[1] != 2:
            log.error(
                "Boundary coordinate array of invalid shape. "
                "Must be numpy.ndarray of size (n, 2)."
            )
            # return False
        if type(exclusions) != list:
            log.error(
                "Exclusion coordinate array of invalid type. " "Cannot initialise mesh."
            )
            # return False
        for exclusion in exclusions:
            if np.shape(exclusion)[1] != 2:
                log.error(
                    "Exclusion coordinate array of invalid shape. "
                    "Must be numpy.ndarray of size (n, 2)."
                )
                # return False

        # Store variables.
        self._image_folder = image_folder
        self._common_file_name = os.path.commonprefix(os.listdir(image_folder)).rstrip(
            "0123456789"
        )
        self._image_indices = np.asarray(
            sorted([int(re.findall(r"\d+", x)[-1]) for x in os.listdir(image_folder)])
        )
        self._number_images = np.shape(self._image_indices)[0]
        self._image_file_type = image_file_type
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
            "common_file_name": self._common_file_name,
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
        track=False,
        seed_coord=None,
        template=None,
        max_iterations=15,
        max_norm=1e-3,
        adaptive_iterations=0,
        method="ICGN",
        order=1,
        tolerance=0.7,
        alpha=0.5,
        mesh_save=False,
    ):
        # Check inputs.
        if type(seed_coord) != np.ndarray:
            try:
                seed_coord = np.asarray(seed_coord)
            except Exception:
                log.error(
                    "Seed coordinate is not of numpy.ndarray type. "
                    "Cannot initiate solver."
                )
                return False
        elif type(adaptive_iterations) != int:
            log.error(
                "Number of adaptive iterations of invalid type. "
                "Must be an integer greater than or equal to zero."
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

        # Store variables.
        self._seed_coord = seed_coord
        self._template = template
        self._max_iterations = max_iterations
        self._max_norm = max_norm
        self._adaptive_iterations = adaptive_iterations
        self._method = method
        self._order = order
        self._tolerance = tolerance
        self._alpha = alpha
        self._p_0 = np.zeros(6 * self._order)

        # Solve.
        _f_index = 0
        _g_index = 1
        _f_img = gp.image.Image(
            self._image_folder
            + "/"
            + self._common_file_name
            + str(self._image_indices[_f_index])
            + self._image_file_type
        )
        _g_img = gp.image.Image(
            self._image_folder
            + "/"
            + self._common_file_name
            + str(self._image_indices[_g_index])
            + self._image_file_type
        )
        while _g_index < len(self._image_indices - 1):
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
            )  # Initialise mesh object.
            mesh.solve(
                seed_coord=self._seed_coord,
                template=self._template,
                max_iterations=self._max_iterations,
                max_norm=self._max_norm,
                adaptive_iterations=self._adaptive_iterations,
                method=self._method,
                order=self._order,
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
                self._boundary_tags = mesh._boundary_tags
                self._exclusion_tags = mesh._exclusion_tags
                _g_index += 1  # Iterate the target image index.
                del _g_img
                if _g_index != len(self._image_indices - 1):
                    _g_img = gp.image.Image(
                        self._image_folder
                        + "/"
                        + self._common_file_name
                        + str(self._image_indices[_g_index])
                        + self._image_file_type
                    )
                else:
                    self.solved = True
            elif _f_index + 1 < _g_index:
                _f_index = _g_index - 1
                if track:
                    self._track(_f_index)
                del _f_img
                _f_img = gp.image.Image(
                    self._image_folder
                    + "/"
                    + self._common_file_name
                    + str(self._image_indices[_f_index])
                    + self._image_file_type
                )
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

    def _track(self, _f_index):
        """
        Private method for tracking the movement of the mesh boundary and exclusions upon reference image updates.
        """

        log.info("Tracing boundary and exclusion displacements.")
        self._boundary = self.data["meshes"][_f_index-1]["nodes"][self._boundary_tags] + self.data["meshes"][_f_index-1]["results"]["displacements"][self._boundary_tags]
        _exclusions = []
        for i in range(len(self._exclusions)):
            _exclusions.append(self.data["meshes"][_f_index-1]["nodes"][self._exclusion_tags[i]] + self.data["meshes"][_f_index-1]["results"]["displacements"][self._exclusion_tags[i]])
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
