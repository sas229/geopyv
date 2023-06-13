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
        self, shape=None, coord=None, boundary=None, rigid=False, hard=True, track=True
    ):
        """
        hard_boundary : bool, optional
            Boolean to control whether the boundary is included in the
            binary mask. True -included, False - not included.
            Defaults to True.
        """
        self._shape = shape
        self._coord = coord
        self._boundary = boundary
        self._rigid = rigid
        self._hard = hard
        self._track = track
        self.solved = False

        self.data = {
            "type": "geometry.Region",
            "solved": self.solved,
            "shape": self._shape,
            "rigid": self._rigid,
            "hard": self._hard,
            "track": self._track,
            "specifics": self._specifics,
            "coords": [self._coord],
            "boundaries": [self._boundary],
        }
        self._reference_update_register = []
        self._ref_index = None

    def _update(self, f_img, warp):
        if self._track:
            if self._rigid:
                local_coordinates = self._boundary - self._coord
                theta = (warp[3] - warp[4]) / 2
                rot = np.asarray(
                    [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
                )
                self.data["boundaries"].append(
                    self._coord + warp[:2] + local_coordinates @ rot
                )
                self.data["coords"].append(self._coord + warp[:2])
            else:
                self.data["boundaries"].append(self._boundary + warp)
                self.data["coords"].append(self._coord + np.mean(warp, axis=0))
            if self._ref_update(f_img):
                self._boundary = self.data["boundaries"][-1]
                self._coord = self.data["coords"][-1]
        self._solved = True
        self.data["solved"] = self._solved

    def _ref_update(self, f_img):
        index = int(
            re.findall(
                r"\d+",
                f_img,
            )[-1]
        )
        print(index, self._ref_index)
        if index != self._ref_index:
            print("Triggered!")
            self._ref_index = index
            return True
        else:
            return False


class Circle(Region):
    """

    Circular Region template class.

    """

    def __init__(
        self, coord=None, radius=50.0, size=20.0, rigid=True, hard=True, track=True
    ):
        """

        Class for circular Region. Subclassed from Region.

        Parameters
        ----------
        coord : numpy.ndarray(2,)
            Central coordinate of the circular Region.

        Attributes
        ----------

        """
        # Input check.
        if self._report(gp.check._check_type(coord, "coord", [np.ndarray]), "Warning"):
            image = gp.io._load_f_img()
            selector = gp.gui.selectors.coordinate.CoordinateSelector()
            coord = selector.select(image, gp.templates.Circle(50))
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
        self._report(gp.check._check_type(track, "track", [bool]), "TypeError")

        # Store unique variables.
        self._size = size
        self._radius = radius
        self._specifics = {
            "size": self._size,
            "radius": self._radius,
        }

        # Defining boundary.
        number_points = np.maximum(
            6, int(2 * np.pi * radius / size)
        )  # Use a minimum of six points irrespective of target size defined.
        theta = np.linspace(0, 2 * np.pi, number_points, endpoint=False)
        x = radius * np.cos(theta) + coord[0]
        y = radius * np.sin(theta) + coord[1]
        boundary = np.column_stack((x, y))

        # Store general variables.
        super().__init__(
            shape="Circle",
            coord=coord,
            boundary=boundary,
            rigid=rigid,
            hard=hard,
            track=track,
        )


class Path(Region):
    """

    Circular Region template class.

    """

    def __init__(self, coord=None, boundary=None, rigid=True, hard=True, track=True):
        """

        Class for circular Region. Subclassed from Region.

        Parameters
        ----------
        coord : numpy.ndarray(2,)
            Central coordinate of the circular Region.

        Attributes
        ---------

        """
        # Input check.
        if self._report(
            gp.check._check_type(coord, "coord", [np.ndarray, type(None)]), "Warning"
        ):
            image = gp.io._load_f_img()
            selector = gp.gui.selectors.coordinate.CoordinateSelector()
            coord = selector.select(image, gp.templates.Circle(50))
        check = gp.check._check_type(boundary, "boundary", [np.ndarray])
        if check:
            try:
                boundary = np.asarray(boundary)
                self._report(
                    gp.check._conversion(boundary, "boundary", np.ndarray, False),
                    "Warning",
                )
            except Exception:
                self._report(check, "TypeError")
        self._report(gp.check._check_dim(boundary, "boundary", 2), "ValueError")
        self._report(gp.check._check_axis(boundary, "boundary", 1, [2]), "ValueError")

        if coord is None:
            coord = np.mean(boundary, axis=0)

        # Store unique variables.
        self._specifics = {}

        # Store general variables.
        super().__init__(
            shape="Path",
            coord=coord,
            boundary=boundary,
            rigid=rigid,
            hard=hard,
            track=track,
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
