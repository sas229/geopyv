import numpy as np
import gmsh

# from scipy.optimize import minimize_scalar
import PIL.Image as ImagePIL
import PIL.ImageDraw as ImageDrawPIL


def _gmsh_initializer():
    # Initialize gmsh if not already initialized.
    if gmsh.isInitialized() == 0:
        gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)


# def _set_target_nodes(target):
#     _gmsh_initializer()
#     log.info(
#         "Generating mesh using gmsh with approximately {n} nodes.".format(
#             n=self._target_nodes
#         )
#     )
#     _initial_mesh()


def _mask_image(f_img, boundary, exclusions):
    """

    Private method to create an image mask.

    """

    binary_img = ImagePIL.new(
        "L",
        (np.shape(f_img.image_gs)[1], np.shape(f_img.image_gs)[0]),
        0,
    )
    if boundary._hard:
        ImageDrawPIL.Draw(binary_img).polygon(
            boundary._nodes.flatten().tolist(), outline=1, fill=1
        )
    else:
        image_edge = np.asarray(
            [
                [0.0, 0.0],
                [0.0, np.shape(f_img.image_gs)[0]],
                [
                    np.shape(f_img.image_gs)[1],
                    np.shape(f_img.image_gs)[0],
                ],
                [np.shape(f_img.image_gs)[1], 0.0],
            ]
        )
        ImageDrawPIL.Draw(binary_img).polygon(
            image_edge.flatten().tolist(), outline=1, fill=1
        )
    # Add exclusion to binary mask.
    for exclusion in exclusions:
        ImageDrawPIL.Draw(binary_img).polygon(
            exclusion._nodes.flatten().tolist(), outline=1, fill=0
        )
    return np.asarray(binary_img)


def _define_RoI(*, f_img=None, boundary=None, exclusions=None):
    """

    Private method to define the RoI.

    """

    if f_img is not None:
        # Create binary mask.
        mask = _mask_image(f_img, boundary, exclusions)
        # Define nodes in relevant space.
        boundary_nodes = boundary._nodes
        exclusion_nodes = [exclusions[i]._nodes for i in range(len(exclusions))]
    else:
        boundary_nodes = boundary
        exclusion_nodes = exclusions
        # try:
        #     boundary_nodes = boundary.data["Nodes"][0]
        #     exclusion_nodes = [
        #         exclusions[i].data["Nodes"][0] for i in range(len(exclusions))
        #     ]
        # except Exception:
        #     boundary_nodes = boundary.data["nodes"][0]
        #     exclusion_nodes = [
        #         exclusions[i].data["nodes"][0] for i in range(len(exclusions))
        #     ]

    # Create objects for mesh generation.
    segments = np.empty(
        (np.shape(boundary_nodes)[0], 2), dtype=np.int32
    )  # Initiate segment array.
    segments[:, 0] = np.arange(
        np.shape(boundary_nodes)[0], dtype=np.int32
    )  # Fill segment array.
    segments[:, 1] = np.roll(segments[:, 0], -1)  # Fill segment array.
    curves = [list(segments[:, 0])]  # Create curve list.

    # Add exclusions.
    borders = boundary_nodes
    for en in exclusion_nodes:
        cur_max_idx = np.amax(segments)  # Highest index used by current segments.
        exclusion_segment = np.empty(np.shape(en))  # Initiate exclusion segment array.
        exclusion_segment[:, 0] = np.arange(
            cur_max_idx + 1, cur_max_idx + 1 + np.shape(en)[0]
        )  # Fill exclusion segment array.
        exclusion_segment[:, 1] = np.roll(
            exclusion_segment[:, 0], -1
        )  # Fill exclusion segment array.
        borders = np.append(borders, en, axis=0)  # Append exclusion to boundary array.
        segments = np.append(segments, exclusion_segment, axis=0).astype(
            "int32"
        )  # Append exclusion segments to segment array.
        curves.append(
            list(exclusion_segment[:, 0].astype("int32"))
        )  # Append exclusion curve to curve list.

    if f_img is not None:
        return borders, segments, curves, mask
    else:
        return borders, segments, curves
