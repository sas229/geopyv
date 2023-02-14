
import numpy as np
import scipy as sp
import gmsh
from scipy.optimize import minimize_scalar
import PIL.Image as ImagePIL
import PIL.ImageDraw as ImageDrawPIL

def _gmsh_initializer():
    # Initialize gmsh if not already initialized.
    if gmsh.isInitialized() == 0:
        gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)


def _set_target_nodes(target):

    _gmsh_initializer()
    log.info("Generating mesh using gmsh with approximately {n} nodes.".format(
                n=self._target_nodes
            )
        )
    _initial_mesh()

    
def _define_RoI(_f_img, _boundary, _exclusions):
        """

        Private method to define the RoI.

        """

        # Create binary mask RoI.
        binary_img = ImagePIL.new(
            "L",
            (np.shape(_f_img.image_gs)[1], np.shape(_f_img.image_gs)[0]),
            0,
        )
        ImageDrawPIL.Draw(binary_img).polygon(
            _boundary.flatten().tolist(), outline=1, fill=1
        )

        # Create objects for mesh generation.
        _segments = np.empty(
            (np.shape(_boundary)[0], 2), dtype=np.int32
        )  # Initiate segment array.
        _segments[:, 0] = np.arange(
            np.shape(_boundary)[0], dtype=np.int32
        )  # Fill segment array.
        _segments[:, 1] = np.roll(_segments[:, 0], -1)  # Fill segment array.
        _curves = [list(_segments[:, 0])]  # Create curve list.

        # Add exclusions.
        for exclusion in _exclusions:
            ImageDrawPIL.Draw(binary_img).polygon(
                exclusion.flatten().tolist(), outline=1, fill=0
            )  # Add exclusion to binary mask.
            cur_max_idx = np.amax(
                _segments
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
            _boundary = np.append(
                _boundary, exclusion, axis=0
            )  # Append exclusion to boundary array.
            _segments = np.append(
                _segments, exclusion_segment, axis=0
            ).astype(
                "int32"
            )  # Append exclusion segments to segment array.
            _curves.append(
                list(exclusion_segment[:, 0].astype("int32"))
            )  # Append exclusion curve to curve list.

        # Finalise mask.
        _mask = np.array(binary_img)
        
        return _boundary, _segments, _curves, _mask