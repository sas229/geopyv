import numpy as np
from geopyv.templates import Circle
from geopyv.image import Image
from geopyv.subset import Subset
from geopyv.io import save, load
from pathlib import Path
import os

# Find test images.
path = Path(os.getcwd())
tests_path = os.path.realpath(__file__)
img_path = Path(tests_path).parents[1]
ref_path = os.path.join(path, img_path, "ref.jpg")
tar_path = os.path.join(path, img_path, "tar.jpg")
ref = Image(ref_path)
tar = Image(tar_path)

# Settings.
max_norm = 1e-3
max_iterations = 50
order = 1
tolerance = 0.7
template = Circle(25)
coord = np.asarray((200.43, 200.76))

# Solve subset.
subset = Subset(f_coord=coord, f_img=ref, g_img=tar, template=template)
success = subset.solve(
    max_norm=max_norm,
    max_iterations=max_iterations,
    order=order,
    tolerance=tolerance,
    method="FAGN",
)


def test_io_save():
    """

    Integration test for geopy.io.save.

    """
    success = save(subset, "test")
    assert success is True


def test_io_save_wrong_type():
    """

    Integration test for geopy.io.save wrong type.

    """
    success = save(None, "test")
    assert success is False


def test_io_load():
    """

    Integration test for geopy.io.load.

    """
    object = load("test")
    assert object is not None


def test_io_load_file_not_found():
    """

    Integration test for geopy.io.load file not found.

    """
    object = load("")
    assert object is None
