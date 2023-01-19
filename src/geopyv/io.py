import pickle
from geopyv.subset import Subset
from geopyv.mesh import Mesh
from geopyv.particle import Particle

def load(filename=None):
    ext = ".pyv"
    filepath = filename + ext
    with open(filepath, "rb") as infile:
        return pickle.load(infile)

def save(object, filename):
    """Function to save data from a geopyv object."""
    if type(object) == Subset or type(object) == Mesh or type(object == Particle):
        object.save(filename) 
