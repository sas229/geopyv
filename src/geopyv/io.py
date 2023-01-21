import logging
import json
import numpy as np
from numpyencoder import NumpyEncoder
from geopyv.subset import Subset, SubsetResults
from geopyv.mesh import Mesh, MeshResults
from geopyv.particle import Particle

log = logging.getLogger(__name__)

def load(filename=None):
    """Function to load a geopyv data file."""
    ext = ".pyv"
    filepath = filename + ext
    try:
        with open(filepath, "r") as infile:
            data =  json.load(infile)
            data = _convert_list_to_ndarray(data)
            if data["type"] == "Subset":
                return SubsetResults(data)
            elif data["type"] == "Mesh":
                return MeshResults(data)
    except:
        raise FileNotFoundError("File not found.")

def save(object, filename):
    """Function to save data from a geopyv object."""
    if type(object) == Subset or type(object) == Mesh or type(object == Particle):
        solved = object.data["solved"]
        if solved == True:
            ext = ".pyv"
            filepath = filename + ext
            with open(filepath, "w") as outfile:
                json.dump(object.data, outfile, cls=NumpyEncoder)
        else:
            log.warn("geopyv object not solved therefore no data to save.")
    else:
        raise TypeError("Not a geopyv type.")

def _convert_list_to_ndarray(data):
    """Recursive function to convert lists to numpy ndarray."""
    for key, value in data.items():
        if type(value) == list:
            data[key] = np.asarray(value)
        elif type(value) == dict:
            _convert_list_to_ndarray(data[key])
    return data