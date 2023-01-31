import numpy as np
from geopyv.templates import Circle
from geopyv.image import Image
from geopyv.subset import Subset
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


def test_Subset_solve_ICGN_W1():
    """

    Integration test for Subset.solve() for first order ICGN.

    """
    max_norm = 1e-3
    max_iterations = 50
    order = 1
    tolerance = 0.7
    template = Circle(25)
    coord = np.asarray((200.43, 200.76))
    subset = Subset(f_coord=coord, f_img=ref, g_img=tar, template=template)
    success = subset.solve(
        max_norm=max_norm,
        max_iterations=max_iterations,
        order=order,
        tolerance=tolerance,
        method="FAGN",
    )
    assert success is True


def test_Subset_solve_FAGN_W1():
    """

    Integration test for Subset.solve() for first order FAGN.

    """
    max_norm = 1e-3
    max_iterations = 50
    order = 1
    tolerance = 0.7
    template = Circle(25)
    coord = np.asarray((200.43, 200.76))
    subset = Subset(f_coord=coord, f_img=ref, g_img=tar, template=template)
    success = subset.solve(
        max_norm=max_norm,
        max_iterations=max_iterations,
        order=order,
        tolerance=tolerance,
        method="FAGN",
    )
    assert success is True


def test_Subset_solve_ICGN_W2():
    """

    Integration test for Subset.solve() for second order ICGN.

    """
    max_norm = 1e-3
    max_iterations = 50
    order = 2
    tolerance = 0.7
    template = Circle(25)
    coord = np.asarray((200.43, 200.76))
    subset = Subset(f_coord=coord, f_img=ref, g_img=tar, template=template)
    success = subset.solve(
        max_norm=max_norm,
        max_iterations=max_iterations,
        order=order,
        tolerance=tolerance,
        method="ICGN",
    )
    assert success is True


def test_Subset_solve_FAGN_W2():
    """

    Integration test for Subset.solve() for second order FAGN.

    """
    max_norm = 1e-3
    max_iterations = 50
    order = 2
    tolerance = 0.7
    template = Circle(25)
    coord = np.asarray((200.43, 200.76))
    subset = Subset(f_coord=coord, f_img=ref, g_img=tar, template=template)
    success = subset.solve(
        max_norm=max_norm,
        max_iterations=max_iterations,
        order=order,
        tolerance=tolerance,
        method="FAGN",
    )
    assert success is True


def test_Subset_solve_ICGN_W1_precondition():
    """

    Integration test for Subset.solve() for first order ICGN with preconditioning.

    """
    max_norm = 1e-3
    max_iterations = 50
    order = 1
    tolerance = 0.7
    template = Circle(25)
    coord = np.asarray((200.43, 200.76))
    p_0 = np.asarray([0.01, 0.02, 0, 0, 0, 0])
    subset = Subset(f_coord=coord, f_img=ref, g_img=tar, template=template)
    success = subset.solve(
        max_norm=max_norm,
        max_iterations=max_iterations,
        order=order,
        tolerance=tolerance,
        p_0=p_0,
        method="FAGN",
    )
    assert success is True


def test_Subset_solve_FAGN_W1_precondition():
    """

    Integration test for Subset.solve() for first order FAGN with preconditioning.

    """
    max_norm = 1e-3
    max_iterations = 50
    order = 1
    tolerance = 0.7
    template = Circle(25)
    coord = np.asarray((200.43, 200.76))
    p_0 = np.asarray([0.01, 0.02, 0, 0, 0, 0])
    subset = Subset(f_coord=coord, f_img=ref, g_img=tar, template=template)
    success = subset.solve(
        max_norm=max_norm,
        max_iterations=max_iterations,
        order=order,
        tolerance=tolerance,
        p_0=p_0,
        method="FAGN",
    )
    assert success is True


def test_Subset_solve_ICGN_W2_precondition():
    """

    Integration test for Subset.solve() for second order ICGN with preconditioning.

    """
    max_norm = 1e-3
    max_iterations = 50
    order = 2
    tolerance = 0.7
    template = Circle(25)
    coord = np.asarray((200.43, 200.76))
    p_0 = np.asarray([0.01, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    subset = Subset(f_coord=coord, f_img=ref, g_img=tar, template=template)
    success = subset.solve(
        max_norm=max_norm,
        max_iterations=max_iterations,
        order=order,
        tolerance=tolerance,
        p_0=p_0,
        method="ICGN",
    )
    assert success is True


def test_Subset_solve_FAGN_W2_precondition():
    """

    Integration test for Subset.solve() for second order FAGN with preconditioning.

    """
    max_norm = 1e-3
    max_iterations = 50
    order = 2
    tolerance = 0.7
    template = Circle(25)
    coord = np.asarray((200.43, 200.76))
    p_0 = np.asarray([0.01, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    subset = Subset(f_coord=coord, f_img=ref, g_img=tar, template=template)
    success = subset.solve(
        max_norm=max_norm,
        max_iterations=max_iterations,
        order=order,
        tolerance=tolerance,
        p_0=p_0,
        method="FAGN",
    )
    assert success is True


def test_Subset_negative_max_norm():
    """

    Integration test for Subset.solve() for negative max_norm.

    """
    max_norm = -1e-3
    max_iterations = 50
    order = 1
    tolerance = 0.7
    template = Circle(25)
    coord = np.asarray((200.43, 200.76))
    subset = Subset(f_coord=coord, f_img=ref, g_img=tar, template=template)
    success = subset.solve(
        max_norm=max_norm,
        max_iterations=max_iterations,
        order=order,
        tolerance=tolerance,
        method="ICGN",
    )
    assert success is False


def test_Subset_non_integer_max_iterations():
    """

    Integration test for Subset.solve() for non-integr max_iterations.

    """
    max_norm = 1e-3
    max_iterations = 50.1
    order = 1
    tolerance = 0.7
    template = Circle(25)
    coord = np.asarray((200.43, 200.76))
    subset = Subset(f_coord=coord, f_img=ref, g_img=tar, template=template)
    success = subset.solve(
        max_norm=max_norm,
        max_iterations=max_iterations,
        order=order,
        tolerance=tolerance,
        method="ICGN",
    )
    assert success is False


def test_Subset_negative_max_iterations():
    """

    Integration test for Subset.solve() for negative max_norm.

    """
    max_norm = 1e-3
    max_iterations = -50
    order = 1
    tolerance = 0.7
    template = Circle(25)
    coord = np.asarray((200.43, 200.76))
    subset = Subset(f_coord=coord, f_img=ref, g_img=tar, template=template)
    success = subset.solve(
        max_norm=max_norm,
        max_iterations=max_iterations,
        order=order,
        tolerance=tolerance,
        method="ICGN",
    )
    assert success is False


def test_Subset_invalid_max_iterations():
    """

    Integration test for Subset.solve() for invalid max_norm.

    """
    max_norm = 1e-3
    max_iterations = 0
    order = 1
    tolerance = 0.7
    template = Circle(25)
    coord = np.asarray((200.43, 200.76))
    subset = Subset(f_coord=coord, f_img=ref, g_img=tar, template=template)
    success = subset.solve(
        max_norm=max_norm,
        max_iterations=max_iterations,
        order=order,
        tolerance=tolerance,
        method="ICGN",
    )
    assert success is False


def test_Subset_negative_tolerance():
    """

    Integration test for Subset.solve() for invalid max_norm.

    """
    max_norm = 1e-3
    max_iterations = 50
    order = 1
    tolerance = -0.7
    template = Circle(25)
    coord = np.asarray((200.43, 200.76))
    subset = Subset(f_coord=coord, f_img=ref, g_img=tar, template=template)
    success = subset.solve(
        max_norm=max_norm,
        max_iterations=max_iterations,
        order=order,
        tolerance=tolerance,
        method="ICGN",
    )
    assert success is False


def test_Subset_invalid_tolerance():
    """

    Integration test for Subset.solve() for invalid max_norm.

    """
    max_norm = 1e-3
    max_iterations = 50
    order = 1
    tolerance = 1.1
    template = Circle(25)
    coord = np.asarray((200.43, 200.76))
    subset = Subset(f_coord=coord, f_img=ref, g_img=tar, template=template)
    success = subset.solve(
        max_norm=max_norm,
        max_iterations=max_iterations,
        order=order,
        tolerance=tolerance,
        method="ICGN",
    )
    assert success is False


def test_Subset_invalid_order():
    """

    Integration test for Subset.solve() for invalid max_norm.

    """
    max_norm = 1e-3
    max_iterations = 50
    order = 3
    tolerance = 0.7
    template = Circle(25)
    coord = np.asarray((200.43, 200.76))
    subset = Subset(f_coord=coord, f_img=ref, g_img=tar, template=template)
    success = subset.solve(
        max_norm=max_norm,
        max_iterations=max_iterations,
        order=order,
        tolerance=tolerance,
        method="ICGN",
    )
    assert success is False


def test_Subset_invalid_p_0_shape_W1():
    """

    Integration test for Subset.solve() for invalid max_norm.

    """
    max_norm = 1e-3
    max_iterations = 50
    order = 1
    tolerance = 0.7
    template = Circle(25)
    coord = np.asarray((200.43, 200.76))
    p_0 = np.asarray([0.01, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    subset = Subset(f_coord=coord, f_img=ref, g_img=tar, template=template)
    success = subset.solve(
        max_norm=max_norm,
        max_iterations=max_iterations,
        order=order,
        p_0=p_0,
        tolerance=tolerance,
        method="ICGN",
    )
    assert success is False


def test_Subset_invalid_p_0_shape_W2():
    """

    Integration test for Subset.solve() for invalid max_norm.

    """
    max_norm = 1e-3
    max_iterations = 50
    order = 2
    tolerance = 0.7
    template = Circle(25)
    coord = np.asarray((200.43, 200.76))
    p_0 = np.asarray([0.01, 0.02, 0, 0, 0, 0])
    subset = Subset(f_coord=coord, f_img=ref, g_img=tar, template=template)
    success = subset.solve(
        max_norm=max_norm,
        max_iterations=max_iterations,
        order=order,
        p_0=p_0,
        tolerance=tolerance,
        method="ICGN",
    )
    assert success is False


def test_Subset_invalid_p_0_type():
    """

    Integration test for Subset.solve() for invalid max_norm.

    """
    max_norm = 1e-3
    max_iterations = 50
    order = 2
    tolerance = 0.7
    template = Circle(25)
    coord = np.asarray((200.43, 200.76))
    p_0 = [0.01, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    subset = Subset(f_coord=coord, f_img=ref, g_img=tar, template=template)
    success = subset.solve(
        max_norm=max_norm,
        max_iterations=max_iterations,
        order=order,
        p_0=p_0,
        tolerance=tolerance,
        method="ICGN",
    )
    assert success is False


def test_Subset_failed_solve_iterations():
    """

    Integration test for Subset.solve() for invalid max_norm.

    """
    max_norm = 1e-3
    max_iterations = 1
    order = 2
    tolerance = 0.7
    template = Circle(25)
    coord = np.asarray((200.43, 200.76))
    p_0 = [0.01, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    subset = Subset(f_coord=coord, f_img=ref, g_img=tar, template=template)
    success = subset.solve(
        max_norm=max_norm,
        max_iterations=max_iterations,
        order=order,
        p_0=p_0,
        tolerance=tolerance,
        method="ICGN",
    )
    assert success is False


def test_Subset_failed_solve_correlation():
    """

    Integration test for Subset.solve() for invalid max_norm.

    """
    max_norm = 1e-3
    max_iterations = 15
    order = 2
    tolerance = 0.99
    template = Circle(25)
    coord = np.asarray((200.43, 200.76))
    p_0 = [0.01, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    subset = Subset(f_coord=coord, f_img=ref, g_img=tar, template=template)
    success = subset.solve(
        max_norm=max_norm,
        max_iterations=max_iterations,
        order=order,
        p_0=p_0,
        tolerance=tolerance,
        method="ICGN",
    )
    assert success is False
