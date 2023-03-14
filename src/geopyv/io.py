"""

IO module for geopyv.

"""
import importlib
import logging
import json
import pickle
import os
import numpy as np
import geopyv as gp
from alive_progress import alive_bar

log = logging.getLogger(__name__)


def load(*, filename=None, old_format=False):
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
    object : geopyv.object.Object
        The geopyv data object loaded.


    .. note::
        * Any .pyv object can be loaded with this function.
        * The data object will be loaded into a `ObjectResults` instance where
          `Object` represents the instance type that generated the data. For example,
          data from a `Subset` instance will be loaded into a `SubsetResults` instance.

    """
    directory = os.getcwd()
    if filename is None:
        dialog = gp.gui.selectors.file.FileSelector()
        filepath = dialog.get_path("Select geopyv data file", directory)
    else:
        ext = ".pyv"
        # filepath = filename + ext
        filepath = directory + "/" + filename + ext
    try:
        with open(filepath, "rb") as infile:
            message = "Loading geopyv object"
            with alive_bar(dual_line=True, bar=None, title=message) as bar:
                bar.text = "-> Loading object from {filepath}...".format(
                    filepath=filepath
                )
                if old_format:
                    log.warning(
                        "json file storage deprecated. "
                        "Load and save objects to convert to new format."
                    )
                    data = json.load(infile)
                else:
                    data = pickle.load(infile)
                data = _convert_list_to_ndarray(data)
                bar()
            object_type = data["type"]
            log.info(
                "Loaded {object_type} object from {filepath}.".format(
                    object_type=object_type, filepath=filepath
                )
            )
            class_name = data["type"] + "Results"
            module = importlib.import_module("geopyv." + object_type.lower())
            results_instance = getattr(module, class_name)
            return results_instance(data)
    except Exception:
        log.error(
            "File does not exist at the path supplied:\n{filepath}".format(
                filepath=filepath
            )
        )
        raise FileExistsError


def save(*, object, filename=None):
    """

    Function to save data from a geopyv object. If no filename is
    provided, the host OS default file browser will be used to allow
    the user to choose a filename and storage location.


    Parameters
    ----------
    object : geopyv.object.Object
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
    if isinstance(object, gp.object.Object):
        solved = object.data["solved"]
        if solved is True:
            ext = ".pyv"
            filepath = directory + "/" + filename + ext
            with open(filepath, "wb") as outfile:
                message = "Saving geopyv object"
                with alive_bar(dual_line=True, bar=None, title=message) as bar:
                    bar.text = "-> Saving object to {filepath}...".format(
                        filepath=filepath
                    )
                    pickle.dump(object.data, outfile)
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
    else:
        log.error("Nothing saved. Object supplied not a valid geopyv type.")
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
        if key == "meshes":
            try:
                for subset in value:
                    _convert_list_to_ndarray(subset)
            except Exception:
                data[key] = list(value)
        # If not a list of subsets, convert to numpy ndarray.
        elif (
            type(value) == list
            and key != "subsets"
            and key != "mesh"
            and key != "particles"
            and key != "field"
        ):
            data[key] = np.asarray(value)
        # If a dict convert recursively.
        elif type(value) == dict:
            _convert_list_to_ndarray(data[key])
        # If a list of subsets convert recursively.
        elif key == "subsets" or key == "mesh" or key == "particles" or key == "field":
            for subset in value:
                _convert_list_to_ndarray(subset)
    return data


def _load_img(message):
    """

    Private method to open a file dialog and select an image.

    """
    directory = os.getcwd()
    dialog = gp.gui.selectors.image.ImageSelector()
    imgpath = dialog.get_path(directory, message)
    img = gp.image.Image(imgpath)
    return img


def _load_f_img():
    """
    Private method to load the reference image.
    """
    log.warn("No reference image supplied. Please select the reference image.")
    return _load_img("Select reference image.")


def _load_g_img():
    """
    Private method to load the target image.
    """
    log.warn("No target image supplied. Please select the target image.")
    return _load_img("Select target image.")


def _get_folder(message):
    """

    Private method to open a file dialog and select a folder.

    """
    directory = os.getcwd()
    dialog = gp.gui.selectors.folder.FolderSelector()
    folder_path = dialog.get_path(directory, message)
    return folder_path


def _get_image_folder():
    """

    Private method to get the `image_folder`.

    """
    log.warn("No `image_folder` supplied. Please select the `image_folder`.")
    return _get_folder("Select image folder.")


def _get_mesh_folder():
    """

    Private method to get the `image_folder`.

    """
    log.warn("No `mesh_folder` supplied. Please select the `mesh_folder`.")
    return _get_folder("Select mesh folder.")
