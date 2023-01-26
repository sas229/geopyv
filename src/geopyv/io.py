import logging
import json
import numpy as np
import geopyv as gp
from numpyencoder import NumpyEncoder
from alive_progress import alive_bar

log = logging.getLogger(__name__)

def load(filename=None):
    """Function to load a geopyv data file."""
    ext = ".pyv"
    filepath = filename + ext
    try:
        with open(filepath, "r") as infile:
            message = "Loading geopyv object"
            with alive_bar(dual_line=True, bar=None, title=message) as bar:
                bar.text = "-> Loading object from {filepath}...".format(filepath=filepath)
                data =  json.load(infile)
                data = _convert_list_to_ndarray(data)
                bar()
            object_type = data["type"]
            log.info("Loaded {object_type} object from {filepath}.".format(object_type=object_type, filepath=filepath))
            if data["type"] == "Subset":
                return gp.subset.SubsetResults(data)
            elif data["type"] == "Mesh":
                return gp.mesh.MeshResults(data)
    except:
        raise FileNotFoundError("File not found.")

def save(object, filename):
    """Function to save data from a geopyv object."""
    if type(object) == gp.subset.Subset or type(object) == gp.mesh.Mesh or type(object == gp.particle.Particle):
        solved = object.data["solved"]
        if solved == True:
            ext = ".pyv"
            filepath = filename + ext
            with open(filepath, "w") as outfile:
                message = "Saving geopyv object"
                with alive_bar(dual_line=True, bar=None, title=message) as bar:
                    bar.text = "-> Saving object to {filepath}...".format(filepath=filepath)
                    json.dump(object.data, outfile, cls=NumpyEncoder)
                    bar()
                object_type = object.data["type"]
                log.info("Saved {object_type} object to {filepath}.".format(object_type=object_type, filepath=filepath))
        else:
            log.warn("Object not solved therefore no data to save.")
    else:
        raise TypeError("Not a geopyv type.")

def _convert_list_to_ndarray(data):
    """Recursive function to convert lists back to numpy ndarray."""
    for key, value in data.items():
        # If not a list of subsets, convert to numpy ndarray.
        if type(value) == list and key != "subsets":
            data[key] = np.asarray(value)
        # If a dict convert recursively.
        elif type(value) == dict:
            _convert_list_to_ndarray(data[key])
        # If a list of subsets convert recursively.
        elif key == "subsets":
            for subset in value:
                _convert_list_to_ndarray(subset)
    return data