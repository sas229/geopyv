import pickle
from geopyv.subset import Subset, SubsetResults
from geopyv.mesh import Mesh
from geopyv.particle import Particle

def load(filename=None):
    """Function to load a geopyv data file."""
    ext = ".pyv"
    filepath = filename + ext
    try:
        with open(filepath, "rb") as infile:
            data =  pickle.load(infile)
            return SubsetResults(data)
    except:
        raise FileNotFoundError("File not found.")

def save(object, name):
    """Function to save data from a geopyv object."""
    if type(object) == Subset or type(object) == Mesh or type(object == Particle):
        object.save(name) 
