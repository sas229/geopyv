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

def test_Subset_solve_ICGN_W1():
    """Integration test for Subset.solve() for first order ICGN."""
    max_norm = 1e-3
    max_iterations = 50
    template = Circle(25)
    ref = Image(ref_path)
    tar = Image(tar_path)
    coord = np.asarray((200.43,200.76))
    p_0 = np.zeros((6))
    subset = Subset(coord, ref, tar, template)
    subset.solve(max_norm = max_norm, max_iterations = max_iterations, p_0 = p_0, method = "ICGN")
    assert subset.norm <= max_norm or subset.iterations < max_iterations

def test_Subset_solve_FAGN_W1():
    """Integration test for Subset.solve() for first order FAGN."""
    max_norm = 1e-3
    max_iterations = 50
    template = Circle(25)
    ref = Image(ref_path)
    tar = Image(tar_path)
    coord = np.asarray((200.43,200.76))
    p_0 = np.zeros((6))
    subset = Subset(coord, ref, tar, template)
    subset.solve(max_norm = max_norm, max_iterations = max_iterations, p_0 = p_0, method = "FAGN")
    assert subset.norm <= max_norm or subset.iterations < max_iterations

def test_Subset_solve_ICGN_W2():
    """Integration test for Subset.solve() for second order ICGN."""
    max_norm = 1e-3
    max_iterations = 50
    template = Circle(25)
    ref = Image(ref_path)
    tar = Image(tar_path)
    coord = np.asarray((200.43,200.76))
    p_0 = np.zeros((12))
    subset = Subset(coord, ref, tar, template)
    subset.solve(max_norm = max_norm, max_iterations = max_iterations, p_0 = p_0, method = "ICGN")
    assert subset.norm <= max_norm or subset.iterations < max_iterations

def test_Subset_solve_FAGN_W2():
    """Integration test for Subset.solve() for second order FAGN."""
    max_norm = 1e-3
    max_iterations = 50
    template = Circle(25)
    ref = Image(ref_path)
    tar = Image(tar_path)
    coord = np.asarray((200.43,200.76))
    p_0 = np.zeros((12))
    subset = Subset(coord, ref, tar, template)
    subset.solve(max_norm = max_norm, max_iterations = max_iterations, p_0 = p_0, method = "FAGN")
    assert subset.norm <= max_norm or subset.iterations < max_iterations

# def test_Subset_solve_WFAGN_W1():
#     """Integration test for Subset.solve() for first order WFAGN."""
#     max_norm = 1e-3
#     max_iterations = 50
#     template = Circle(25)
#     ref = Image(ref_path)
#     tar = Image(tar_path)
#     coord = np.asarray((200.43,200.76))
#     p_0 = np.zeros((7))
#     p_0[-1] = 100
#     subset = Subset(coord, ref, tar, template)
#     subset.solve(max_norm = max_norm, max_iterations = max_iterations, p_0 = p_0, method = "WFAGN")
#     assert subset.norm <= max_norm or subset.iterations < max_iterations

def test_Subset_pysolve_ICGN_W1():
    """Integration test for Subset.pysolve() for first order ICGN."""
    max_norm = 1e-3
    max_iterations = 50
    template = Circle(25)
    ref = Image(ref_path)
    tar = Image(tar_path)
    coord = np.asarray((200.43,200.76))
    p_0 = np.zeros((6))
    subset = Subset(coord, ref, tar, template)
    ref_subset = Subset(coord, ref, tar, template)
    subset.pysolve(max_norm = max_norm, max_iterations = max_iterations, p_0 = p_0, method = "ICGN")
    ref_subset.solve(max_norm = max_norm, max_iterations = max_iterations, p_0 = p_0, method = "ICGN")
    assert subset.norm <= max_norm or subset.iterations < max_iterations


def test_Subset_pysolve_FAGN_W1():
    """Integration test for Subset.pysolve() for first order FAGN."""
    max_norm = 1e-3
    max_iterations = 50
    template = Circle(25)
    ref = Image(ref_path)
    tar = Image(tar_path)
    coord = np.asarray((200.43,200.76))
    p_0 = np.zeros((6))
    subset = Subset(coord, ref, tar, template)
    ref_subset = Subset(coord, ref, tar, template)
    subset.pysolve(max_norm = max_norm, max_iterations = max_iterations, p_0 = p_0, method = "FAGN")
    ref_subset.solve(max_norm = max_norm, max_iterations = max_iterations, p_0 = p_0, method = "FAGN")
    assert subset.norm <= max_norm or subset.iterations < max_iterations


# def test_Subset_pysolve_WFAGN_W1():
#     """Integration test for Subset.pysolve() for first order FAGN."""
#     max_norm = 1e-3
#     max_iterations = 50
#     template = Circle(25)
#     ref = Image(ref_path)
#     tar = Image(tar_path)
#     coord = np.asarray((200.43,200.76))
#     p_0 = np.zeros((7))
#     subset = Subset(coord, ref, tar, template)
#     ref_subset = Subset(coord, ref, tar, template)
#     subset.pysolve(max_norm = max_norm, max_iterations = max_iterations, p_0 = p_0, method = "WFAGN")
#     ref_subset.solve(max_norm = max_norm, max_iterations = max_iterations, p_0 = p_0, method = "WFAGN")
#     assert subset.norm <= max_norm or subset.iterations < max_iterations


def test_Subset_pysolve_ICGN_W2():
    """Integration test for Subset.pysolve() for first order ICGN."""
    max_norm = 1e-3
    max_iterations = 50
    template = Circle(25)
    ref = Image(ref_path)
    tar = Image(tar_path)
    coord = np.asarray((200.43,200.76))
    p_0 = np.zeros((12))
    subset = Subset(coord, ref, tar, template)
    ref_subset = Subset(coord, ref, tar, template)
    subset.pysolve(max_norm = max_norm, max_iterations = max_iterations, p_0 = p_0, method = "ICGN")
    ref_subset.solve(max_norm = max_norm, max_iterations = max_iterations, p_0 = p_0, method = "ICGN")
    assert subset.norm <= max_norm or subset.iterations < max_iterations


def test_Subset_pysolve_FAGN_W2():
    """Integration test for Subset.pysolve() for first order FAGN."""
    max_norm = 1e-3
    max_iterations = 50
    template = Circle(25)
    ref = Image(ref_path)
    tar = Image(tar_path)
    coord = np.asarray((200.43,200.76))
    p_0 = np.zeros((12))
    subset = Subset(coord, ref, tar, template)
    ref_subset = Subset(coord, ref, tar, template)
    subset.pysolve(max_norm = max_norm, max_iterations = max_iterations, p_0 = p_0, method = "FAGN")
    ref_subset.solve(max_norm = max_norm, max_iterations = max_iterations, p_0 = p_0, method = "FAGN")
    assert subset.norm <= max_norm or subset.iterations < max_iterations


