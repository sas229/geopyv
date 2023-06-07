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


def _define_RoI(f_img, boundary, exclusions):
    """

    Private method to define the RoI.

    """

    # Create binary mask RoI.
    binary_img = ImagePIL.new(
        "L",
        (np.shape(f_img.image_gs)[1], np.shape(f_img.image_gs)[0]),
        0,
    )
    if boundary._hard:
        ImageDrawPIL.Draw(binary_img).polygon(
            boundary._boundary.flatten().tolist(), outline=1, fill=1
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

    # Create objects for mesh generation.
    segments = np.empty(
        (np.shape(boundary._boundary)[0], 2), dtype=np.int32
    )  # Initiate segment array.
    segments[:, 0] = np.arange(
        np.shape(boundary._boundary)[0], dtype=np.int32
    )  # Fill segment array.
    segments[:, 1] = np.roll(segments[:, 0], -1)  # Fill segment array.
    curves = [list(segments[:, 0])]  # Create curve list.

    # Add exclusions.
    borders = boundary._boundary
    for exclusion in exclusions:
        ImageDrawPIL.Draw(binary_img).polygon(
            exclusion._boundary.flatten().tolist(), outline=1, fill=0
        )  # Add exclusion to binary mask.
        cur_max_idx = np.amax(segments)  # Highest index used by current segments.
        exclusion_segment = np.empty(
            np.shape(exclusion._boundary)
        )  # Initiate exclusion segment array.
        exclusion_segment[:, 0] = np.arange(
            cur_max_idx + 1, cur_max_idx + 1 + np.shape(exclusion._boundary)[0]
        )  # Fill exclusion segment array.
        exclusion_segment[:, 1] = np.roll(
            exclusion_segment[:, 0], -1
        )  # Fill exclusion segment array.
        borders = np.append(
            borders, exclusion._boundary, axis=0
        )  # Append exclusion to boundary array.
        segments = np.append(segments, exclusion_segment, axis=0).astype(
            "int32"
        )  # Append exclusion segments to segment array.
        curves.append(
            list(exclusion_segment[:, 0].astype("int32"))
        )  # Append exclusion curve to curve list.

    # Finalise mask.
    mask = np.array(binary_img)

    return borders, segments, curves, mask
