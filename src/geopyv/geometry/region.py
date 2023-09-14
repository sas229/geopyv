"""

Region module for geopyv.

"""
import logging
import numpy as np
import geopyv as gp
from geopyv.object import Object
import re

log = logging.getLogger(__name__)


class RegionBase(Object):
    """

    Region base class.

    """

    def __init__(self):
        super().__init__(object_type="geometry.Region")
        """

        Mesh base class initialiser.

        """

    def _report(self, msg, error_type):
        if msg and error_type != "Warning":
            log.error(msg)
        elif msg and error_type == "Warning":
            log.warning(msg)
            return True
        if error_type == "ValueError" and msg:
            raise ValueError(msg)
        elif error_type == "TypeError" and msg:
            raise TypeError(msg)
        elif error_type == "IndexError" and msg:
            raise IndexError(msg)


class Region(RegionBase):
    def __init__(
        self,
        *,
        shape=None,
        centre=None,
        nodes=None,
        option="F",
        hard=True,
        compensate=True,
    ):
        """
        hard : bool, optional
            Boolean to control whether the nodes is included in the
            binary mask. True -included, False - not included.
            Defaults to True.
        option :
            "D" : Defined: a controlled region.
            "S" : Static: an untracked region.
            "R" : Rigid: a tracked region, unable to deform.
            "F" : Flexible: a tracked region, able to deform.
        """
        self._shape = shape
        self._option = option
        if self._option == "D":
            centres = [np.mean(nodes[i], axis=0) for i in range(np.shape(nodes)[0])]
            self._nodes = nodes[0]
            self._centre = centres[0]
        else:
            self._nodes = nodes
            self._centre = centre
            nodes = [nodes]
            centres = [centre]
        self._hard = hard
        self._compensate = compensate

        self.solved = False

        self.data = {
            "type": "geometry.Region",
            "solved": self.solved,
            "shape": self._shape,
            "hard": self._hard,
            "compensate": self._compensate,
            "specifics": self._specifics,
            "centres": centres,
            "nodes": nodes,
            "counter": 0,
            "ref_index": None,
        }
        self._reference_update_register = []

    def _store(self, warp):
        if self._option == "R":
            local_coordinates = self._nodes - self._centre
            theta = (warp[3] - warp[4]) / 2
            rot = np.asarray(
                [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
            )
            self.data["nodes"].append(self._centre + warp[:2] + local_coordinates @ rot)
            self.data["centres"].append(self._centre + warp[:2])
        elif self._option == "F":
            self.data["nodes"].append(self._nodes + warp)
            self.data["centres"].append(self._centre + np.mean(warp, axis=0))
        self._solved = True
        self.data["solved"] = self._solved
        self.data["counter"] += 1

    def _update(self, f_img_filepath):
        if self._option != "S":
            f_index = int(
                re.findall(
                    r"\d+",
                    f_img_filepath,
                )[-1]
            )
            if f_index != self.data["ref_index"]:
                self.data["ref_index"] = f_index
                self._reference_update_register.append(f_index)
                self._nodes = self.data["nodes"][self.data["counter"]]
                self._centre = self.data["centres"][self.data["counter"]]


class Circle(Region):
    """

    Circular Region template class.

    """

    def __init__(
        self,
        *,
        centre=None,
        radius=50.0,
        size=20.0,
        option="F",
        hard=True,
        compensate=True,
    ):
        """

        Class for circular Region. Subclassed from Region.

        Parameters
        ----------
        centre : numpy.ndarray(2,)
            Central coordinate of the circular Region.

        Attributes
        ----------

        """
        # Input check.
        if self._report(
            gp.check._check_type(centre, "centre", [np.ndarray]), "Warning"
        ):
            image = gp.io._load_f_img()
            selector = gp.gui.selectors.coordinate.coordinateSelector()
            centre = selector.select(image, gp.templates.Circle(50))
        check = gp.check._check_type(radius, "radius", [float])
        if check:
            try:
                radius = float(radius)
                self._report(gp.check._conversion(radius, "radius", float), "Warning")
            except Exception:
                self._report(check, "TypeError")
        self._report(
            gp.check._check_range(radius, "radius", 0),
            "ValueError",
        )
        check = gp.check._check_type(size, "size", [float])
        if check:
            try:
                size = float(size)
                self._report(gp.check._conversion(size, "size", float), "Warning")
            except Exception:
                self._report(check, "TypeError")
        self._report(
            gp.check._check_range(size, "size", 0),
            "ValueError",
        )
        self._report(gp.check._check_type(hard, "hard", [bool]), "TypeError")

        # Store unique variables.
        self._size = size
        self._radius = radius
        self._specifics = {
            "size": self._size,
            "radius": self._radius,
        }

        # Defining nodes.
        number_points = np.maximum(
            6, int(2 * np.pi * radius / size)
        )  # Use a minimum of six points irrespective of target size defined.
        theta = np.linspace(0, 2 * np.pi, number_points, endpoint=False)
        x = radius * np.cos(theta) + centre[0]
        y = radius * np.sin(theta) + centre[1]
        nodes = np.column_stack((x, y))

        # Store general variables.
        super().__init__(
            shape="Circle",
            centre=centre,
            nodes=nodes,
            option=option,
            hard=hard,
            compensate=compensate,
        )


class Path(Region):
    """

    Path Region template class.

    """

    def __init__(
        self,
        *,
        centre=None,
        nodes=None,
        option="F",
        hard=True,
        compensate=True,
        radius=25,
    ):
        """

        Class for circular Region. Subclassed from Region.

        Parameters
        ----------
        centre : numpy.ndarray(2,)
            Central coordinate of the circular Region.

        Attributes
        ---------

        """
        # Input check.
        if self._report(
            gp.check._check_type(centre, "centre", [np.ndarray, type(None)]), "Warning"
        ):
            image = gp.io._load_f_img()
            selector = gp.gui.selectors.coordinate.coordinateSelector()
            centre = selector.select(image, gp.templates.Circle(50))
        check = gp.check._check_type(nodes, "nodes", [np.ndarray])
        if check:
            try:
                nodes = np.asarray(nodes)
                self._report(
                    gp.check._conversion(nodes, "nodes", np.ndarray, False),
                    "Warning",
                )
            except Exception:
                self._report(check, "TypeError")
        # self._report(gp.check._check_dim(nodes, "nodes", 2), "ValueError")
        # self._report(gp.check._check_axis(nodes, "nodes", 1, [2]), "ValueError")

        if centre is None:
            centre = np.mean(nodes, axis=0)

        # Store unique variables.
        self._specifics = {"radius": radius}

        # Store general variables.
        super().__init__(
            shape="Path",
            centre=centre,
            nodes=nodes,
            option=option,
            hard=hard,
            compensate=compensate,
        )


class RegionResults(RegionBase):
    """
    RegionResults class for geopyv.

    Parameters
    ----------
    data : dict
        geopyv data dict from Region object.

    Attributes
    ----------
    data : dict
        geopyv data dict from Region object.

    """

    def __init__(self, data):
        """Initialisation of geopyv RegionResults class."""
        self.data = data
