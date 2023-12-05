"""

IO module for geopyv.

"""
import importlib
import logging
import pickle
import os
import geopyv as gp
from alive_progress import alive_bar

log = logging.getLogger(__name__)


def load(*, filename=None, directory=None, old_format=False, verbose=True):
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
    cwd = os.getcwd()
    if directory is None and cwd not in filename:
        directory = os.getcwd()
    if filename is None:
        dialog = gp.gui.selectors.file.FileSelector()
        filepath = dialog.get_path("Select geopyv data file", directory)
    else:
        ext = ".pyv"
        # filepath = filename + ext
        if ext in filename:
            filepath = directory + "/" + filename
        else:
            filepath = directory + "/" + filename + ext
    try:
        with open(filepath, "rb") as infile:
            if verbose:
                message = "Loading geopyv object"
                with alive_bar(dual_line=True, bar=None, title=message) as bar:
                    bar.text = "-> Loading object from {filepath}...".format(
                        filepath=filepath
                    )
                    data = pickle.load(infile)
                    bar()
            else:
                data = pickle.load(infile)
            object_type = data["type"]
            if verbose:
                log.info(
                    "Loaded {object_type} object from {filepath}.".format(
                        object_type=object_type, filepath=filepath
                    )
                )
            if "." in data["type"]:
                class_name = data["type"].split(".")[-1] + "Results"
            else:
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


def save(*, object, directory=None, filename=None, verbose=True):
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
    cwd = os.getcwd()
    if directory is None and cwd not in filename:
        directory = os.getcwd()
    if filename is None:
        log.error(
            "No filename provided."
        )  # Add a method to select the filename here...
        return False
    if isinstance(object, gp.object.Object):
        solved = object.data["solved"]
        if solved is True or object.data["type"] == "Sequence":
            if filename[-4:] == ".pyv":
                filename = filename[:-4]
            ext = ".pyv"
            filepath = directory + "/" + filename + ext
            with open(filepath, "wb") as outfile:
                if verbose:
                    message = "Saving geopyv object"
                    with alive_bar(dual_line=True, bar=None, title=message) as bar:
                        bar.text = "-> Saving object to {filepath}...".format(
                            filepath=filepath
                        )
                        pickle.dump(object.data, outfile)
                        bar()
                else:
                    pickle.dump(object.data, outfile)
                object_type = object.data["type"]
                if verbose:
                    log.info(
                        "Saved {object_type} object to {filepath}.".format(
                            object_type=object_type, filepath=filepath
                        )
                    )
                return True
        else:
            log.warn(
                "`{}` object not solved therefore no data to save.".format(
                    object.data["type"]
                )
            )
            return False
    else:
        log.error("Nothing saved. Object supplied not a valid geopyv type.")
        return False


def _img_loader(message, load):
    """

    Private method to open a file dialog and select an image.

    """
    directory = os.getcwd()
    dialog = gp.gui.selectors.image.ImageSelector()
    imgpath = dialog.get_path(directory, message)
    if load:
        return gp.image.Image(imgpath)
    else:
        return imgpath


def _load_f_img():
    """
    Private method to load the reference image.
    """
    log.warn("No reference image supplied. Please select the reference image.")
    return _img_loader("Select reference image.")


def _load_g_img():
    """
    Private method to load the target image.
    """
    log.warn("No target image supplied. Please select the target image.")
    return _img_loader("Select target image.")


def _load_img(name, load=True):
    """
    Private method to load an image.
    """
    log.warn("No {} image supplied. Please select the {} image.".format(name, name))
    return _img_loader("Select {} image.".format(name), load)


def _get_folder(message):
    """

    Private method to open a file dialog and select a folder.

    """
    directory = os.getcwd()
    dialog = gp.gui.selectors.folder.FolderSelector()
    folder_path = dialog.get_path(directory, message)
    return folder_path[len(directory) :]


def _get_image_dir():
    """

    Private method to get the `image_dir`.

    """
    log.warn("No `image_dir` supplied. Please select the `image_dir`.")
    return _get_folder("Select image directory.")


def _get_mesh_dir():
    """

    Private method to get the `mesh_dir`.

    """
    log.warn("No `mesh_folder` supplied. Please select the `mesh_folder`.")
    return _get_folder("Select mesh directory.")
