import logging
import json
import os
import numpy as np
import geopyv as gp
from numpyencoder import NumpyEncoder
from alive_progress import alive_bar

log = logging.getLogger(__name__)


def load(filename=None):
    """

    Function to load a geopyv data object into the workspace. If no filename
     is provided, the host OS default file browser will be used to allow the
      user to select a geopyv data file with .pyv extension.


    Parameters
    ----------
    filename : str, optional
        Name of file containing a geopy data object.


    Returns
    -------
    object : geopyv.object
        The geopyv data object loaded.


    .. note::
        * Any .pyv object can be loaded with this function.
        * The data object will be loaded into a `ObjectResults` instance where
          `Object` represents the instance type that generated the data. For example,
          data from a `Subset` instance will be loaded into a `SubsetResults` instance.

    """
    if filename is None:
        directory = os.getcwd()
        dialog = gp.gui.selectors.file.FileSelector()
        filepath = dialog.get_path("Select geopyv data file", directory)
    else:
        ext = ".pyv"
        filepath = filename + ext
    try:
        with open(filepath, "r") as infile:
            message = "Loading geopyv object"
            with alive_bar(dual_line=True, bar=None, title=message) as bar:
                bar.text = "-> Loading object from {filepath}...".format(
                    filepath=filepath
                )
                data = json.load(infile)
                data = _convert_list_to_ndarray(data)
                bar()
            object_type = data["type"]
            log.info(
                "Loaded {object_type} object from {filepath}.".format(
                    object_type=object_type, filepath=filepath
                )
            )
            if data["type"] == "Subset":
                return gp.subset.SubsetResults(data)
            elif data["type"] == "Mesh":
                return gp.mesh.MeshResults(data)
    except Exception:
        log.error("File not found.")
        return None


def save(object, filename=None):
    """

    Function to save data from a geopyv object. If no filename is
    provided, the host OS default file browser will be used to allow
    the user to choose a filename and storage location.


    Parameters
    ----------
    object : geopyv.object
        The object to be saved.
    filename : str, optional
        The filename to give to the saved data file.


    .. note::
        * Any geopyv object can be passed to this function.
        * Do not include the .pyv extension in the `filename` argument.

    """
    directory = os.getcwd()
    if filename is None:
        log.error(
            "No filename provided."
        )  # Add a method to select the filename here...
        return False
    if (
        type(object) != gp.subset.Subset
        and type(object) != gp.mesh.Mesh
        and type(object != gp.particle.Particle)
    ):
        log.error("Nothing saved. Object supplied not a valid geopyv type.")
        return False
    if (
        type(object) == gp.subset.Subset
        or type(object) == gp.mesh.Mesh
        or type(object == gp.particle.Particle)
    ):
        solved = object.data["solved"]
        if solved is True:
            ext = ".pyv"
            filepath = directory + "/" + filename + ext
            with open(filepath, "w") as outfile:
                message = "Saving geopyv object"
                with alive_bar(dual_line=True, bar=None, title=message) as bar:
                    bar.text = "-> Saving object to {filepath}...".format(
                        filepath=filepath
                    )
                    json.dump(object.data, outfile, cls=NumpyEncoder)
                    bar()
                object_type = object.data["type"]
                log.info(
                    "Saved {object_type} object to {filepath}.".format(
                        object_type=object_type, filepath=filepath
                    )
                )
                return True
        else:
            log.warn("Object not solved therefore no data to save.")
            return False


def _convert_list_to_ndarray(data):
    """

    Recursive function to convert lists back to numpy ndarray.


    Parameters
    ----------
    data : dict
        Data dictionary containing lists to be parsed into `numpy.ndarray`.


    Returns
    -------
    data : dict
        Data dictionary containing `numpy.ndarray` after conversion from lists.


    """
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
