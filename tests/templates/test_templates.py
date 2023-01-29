from geopyv.templates import Template, Circle, Square
import numpy as np


def test_type_conversion():
    """Check size parameter is converted to an integer."""
    size = 49.99
    template = Template(size)
    assert type(template.size) == int


def test_sign_check():
    """Check size parameter sign is converted to a positive value."""
    size = -25
    template = Template(size)
    assert template.size == np.abs(-25)


def test_Circle_output_shape():
    """Check output shape is correct."""
    size = 25
    template = Circle(size)
    rows = template.coords.shape[0]
    cols = template.coords.shape[1]
    assert rows > 0 and cols == 2


def test_Circle_output_range():
    """Check output ranges are not less than or greater than one radius."""
    size = 25
    template = Circle(size)
    min_coords = np.min(template.coords)
    max_coords = np.max(template.coords)
    assert min_coords <= template.size and max_coords <= template.size

    
def test_Circle_output_type():
    """Check coords output is a numpy array."""
    size = 25
    template = Circle(size)
    assert type(template.coords) == np.ndarray


def test_Square_output_shape():
    """Check output shape is correct."""
    size = 25
    template = Square(size)
    rows = template.coords.shape[0]
    cols = template.coords.shape[1]
    assert rows == 2601 and cols == 2


def test_Square_output_range():
    """Check output ranges are not less than or greater than one length."""
    size = 25
    template = Square(size)
    min_coords = np.min(template.coords)
    max_coords = np.max(template.coords)
    assert min_coords <= template.size and max_coords <= template.size


def test_Square_output_type():
    """Check coords output is a numpy array."""
    size = 25
    template = Square(size)
    assert type(template.coords) == np.ndarray

# def test_Circle_mask():
#     """Check subset mask relative to a boundary and a hole."""
#     size = 25
#     centre = np.asarray([0.0,0.0])
#     boundary = np.asarray([[-26.0,-26.0],[-26.0,26.0],[26.0,26.0],[26.0,-26.0]])

#     template = Circle(size)
#     template.mask(centre, boundary)
#     assert len(template.coords)==len(Circle(size).coords)

#     centre = np.asarray([26.0,0.0])
#     template = Circle(size)
#     template.mask(centre, boundary)
#     assert (template.coords[:,0]<=0.0).all()

#     centre = np.asarray([0.0,26.0])
#     template = Circle(size)
#     template.mask(centre, boundary)
#     assert (template.coords[:,1]<=0.0).all()

#     centre = np.asarray([0.0,0.0])
#     holes = np.asarray([[[0.0,0.0],[26.0,0.0],[26.0,26.0],[0.0,26.0]], 
#                         [[0.0,0.0],[-26.0,0.0],[-26.0,-26.0],[0.0,-26.0]]])
#     template = Circle(size)
#     template.mask(centre, boundary, holes)
#     assert ((template.coords[:,0]*template.coords[:,1])<=0.0).all()