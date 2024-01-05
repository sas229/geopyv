"""

Speckle module for geopyv.

"""

import logging
import geopyv as gp
import numpy as np
from geopyv.object import Object
from PIL import Image as Im
from alive_progress import alive_bar

log = logging.getLogger(__name__)


class SpeckleBase(Object):
    """

    Speckle base class to be used as a mixin. Contains plot functionality.

    """

    def __init__(self):
        super().__init__(object_type="Speckle")
        """

        Speckle base class initialiser.

        """

    def _warp(self, i, ref, rot=False):
        _warp = np.zeros((np.shape(ref)[0], np.shape(self.data["comp"])[0]))
        delta = ref - self.data["origin"]
        if self.data["vars"] is not None:
            if self.data["vars"]["mode"] == "sin":
                b = self.data["vars"]["width"] / 2
                a = b * self.data["pm"][i][4]
                _warp[:, 0] += (
                    (abs(delta[:, 1]) < b)
                    * a
                    * np.sin(np.pi * delta[:, 1] / self.data["vars"]["width"])
                )
                _warp[:, 0] += (delta[:, 1] >= b) * a
                _warp[:, 0] -= (delta[:, 1] <= -b) * a
                _warp[:, 4] += (
                    (abs(delta[:, 1]) < b)
                    * a
                    * np.pi
                    / self.data["vars"]["width"]
                    * np.cos(np.pi * delta[:, 1] / self.data["vars"]["width"])
                )
            elif self.data["vars"]["mode"] == "bend":
                b = self.data["vars"]["width"] / 2
                a = self.data["pm"][i][4]
                _warp[:, 0] += (
                    (abs(delta[:, 1]) < b)
                    * a
                    * self.data["vars"]["width"]
                    / np.pi
                    * np.cos(np.pi * delta[:, 1] / self.data["vars"]["width"])
                )
                _warp[:, 0] += (delta[:, 1] >= b) * a * (b - delta[:, 1])
                _warp[:, 0] += (delta[:, 1] <= -b) * a * (delta[:, 1] + b)
                _warp[:, 4] += (
                    (abs(delta[:, 1]) < b)
                    * a
                    * -np.sin(np.pi * delta[:, 1] / self.data["vars"]["width"])
                )
                _warp[:, 4] += (delta[:, 1] >= b) * -a
                _warp[:, 4] += (delta[:, 1] <= -b) * a
            elif self.data["vars"]["mode"] == "lin":
                b = self.data["vars"]["width"] / 2
                a = b * self.data["pm"][i][4]
                _warp[:, 0] += (
                    (abs(delta[:, 1]) < b) * self.data["pm"][i][4] * delta[:, 1]
                )
                _warp[:, 0] += (delta[:, 1] >= b) * a
                _warp[:, 0] -= (delta[:, 1] <= -b) * a
                _warp[:, 4] += (abs(delta[:, 1]) < b) * self.data["pm"][i][4]
            elif self.data["vars"]["mode"] == "quad":
                _warp[:, 0] += (
                    (delta[:, 1] >= 0) * 0.5 * self.data["pm"][i][4] * delta[:, 1] ** 2
                )
                _warp[:, 4] += (delta[:, 1] >= 0) * self.data["pm"][i][4] * delta[:, 1]
        else:
            _warp[:, 0] = (
                self.data["pm"][i][0]
                + self.data["pm"][i][2] * delta[:, 0]
                + self.data["pm"][i][4] * delta[:, 1]
                + 0.5 * self.data["pm"][i][6] * delta[:, 0] ** 2
                + self.data["pm"][i][8] * delta[:, 0] * delta[:, 1]
                + 0.5 * self.data["pm"][i][10] * delta[:, 1] ** 2
            )
            _warp[:, 1] = (
                self.data["pm"][i][1]
                + self.data["pm"][i][3] * delta[:, 0]
                + self.data["pm"][i][5] * delta[:, 1]
                + 0.5 * self.data["pm"][i][7] * delta[:, 0] ** 2
                + self.data["pm"][i][9] * delta[:, 0] * delta[:, 1]
                + 0.5 * self.data["pm"][i][11] * delta[:, 1] ** 2
            )
            _warp[:, 2] = (
                self.data["pm"][i][2]
                + self.data["pm"][i][6] * delta[:, 0]
                + self.data["pm"][i][8] * delta[:, 1]
            )
            _warp[:, 3] = (
                self.data["pm"][i][3]
                + self.data["pm"][i][7] * delta[:, 0]
                + self.data["pm"][i][9] * delta[:, 1]
            )
            _warp[:, 4] = (
                self.data["pm"][i][4]
                + self.data["pm"][i][8] * delta[:, 1]
                + self.data["pm"][i][10] * delta[:, 1]
            )
            _warp[:, 5] = (
                self.data["pm"][i][5]
                + self.data["pm"][i][9] * delta[:, 1]
                + self.data["pm"][i][11] * delta[:, 1]
            )
            _warp[:, 6:] = self.data["pm"][i][6:]

            if rot:
                _warp[:, :2] -= delta

        return _warp

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


class Speckle(SpeckleBase):
    """
    Speckle synthetic image generator.

    """

    def __init__(
        self,
        *,
        name=None,
        image_dir=None,
        file_format=".jpg",
        image_size_x=1001,
        image_size_y=1001,
        speckle_limits=1001,
        image_no=101,
        mmin=0,
        mmax=1,
        mtyp=0,
        comp=np.zeros(12),
        origin=np.asarray([501.0, 501.0]),
        noise=np.asarray([[0.0, 0.0], [0, 0]]),
        tmi=100,
        speckle_size=10,
    ):
        self._report(gp.check._check_type(name, "name", [str]), "Error")
        self._report(gp.check._check_type(image_dir, "image_dir", [str]), "Error")
        self._report(
            gp.check._check_type(file_format, "file_format", [str]), "TypeError"
        )
        file_format = gp.check._check_character(file_format, ".", 0)
        self._report(
            gp.check._check_value(
                file_format,
                "file_format",
                [".jpg", ".png", ".bmp", ".JPG", ".PNG", ".BMP"],
            ),
            "ValueError",
        )
        if self._report(gp.check._check_path(image_dir, "image_dir"), "Warning"):
            image_dir = gp.io._get_image_dir()
        image_dir = gp.check._check_character(image_dir, "/", -1)
        check = gp.check._check_type(image_no, "image_no", [int])
        if check:
            try:
                image_no = int(image_no)
                self._report(gp.check._conversion(image_no, "image_no", int), "Warning")
            except Exception:
                self._report(check, "TypeError")
        self._report(gp.check._check_range(image_no, "image_no", 1), "ValueError")
        self._report(gp.check._check_type(mmin, "mmin", [float, int]), "Error")
        self._report(gp.check._check_type(mmax, "mmax", [float, int]), "Error")
        self._report(gp.check._check_type(mtyp, "mtyp", [int]), "Error")
        self._report(gp.check._check_type(comp, "comp", [np.ndarray]), "Error")
        self._report(gp.check._check_axis(comp, "comp", 0, [12]), "Error")

        self._name = name
        self._image_dir = image_dir
        self._image_size_x = image_size_x
        self._image_size_y = image_size_y
        self._speckle_limits = speckle_limits
        self._file_format = file_format
        self._image_no = image_no
        self._mmin = mmin
        self._mmax = mmax
        self._mtyp = mtyp
        self._comp = comp
        self._origin = origin
        self._noise = noise
        self._tmi = tmi
        self._speckle_size = speckle_size
        self.solved = False
        self._ref_speckle = None

        # Data.
        self.data = {
            "type": "Speckle",
            "name": self._name,
            "solved": self.solved,
            "image_size": [
                self._image_size_x,
                self._image_size_y,
            ],
            "image_dir": self._image_dir,
            "file_format": self._file_format,
            "image_no": self._image_no,
            "comp": self._comp,
            "origin": self._origin,
            "noise": self._noise,
            "tmi": self._tmi,
            "speckle_size": self._speckle_size,
        }

    def solve(self, *, wrap=False, number=None, generate=True, vars=None):
        self._wrap = wrap
        self._vars = vars
        [self.X, self.Y] = np.meshgrid(
            range(self._image_size_x), range(self._image_size_y)
        )
        if generate:
            self._ref_speckle = self._speckle_distribution(number)
        self._mult()
        self.data.update(
            {
                "wrap": self._wrap,
                "ref_speckle": self._ref_speckle,
                "pm": self._pm,
                "vars": self._vars,
            }
        )
        self._image_generation()
        self.solved = True
        self.data["solved"] = True

    def _speckle_distribution(self, number):
        raw_grid = np.zeros((self._image_size_y, self._image_size_x))

        if number is not None:
            corner = np.asarray(
                [
                    0.5 * (self._image_size_x - self._speckle_limits),
                    0.5 * (self._image_size_y - self._speckle_limits),
                ]
            )
            speckles = corner + np.random.rand(number, 2) * self._speckle_limits
        else:
            i = 0
            mi = 0
            speckles = []
            with alive_bar(
                self._tmi,
                manual=True,
                dual_line=True,
                bar="blocks",
                title="Speckling...",
            ) as bar:
                while mi < self._tmi and i < 50000:
                    corner = np.asarray(
                        [
                            0.5 * self._image_size_y - self._speckle_limits,
                            0.5 * self._image_size_x - self._speckle_limits,
                        ]
                    )
                    new_speckle = corner + np.random.rand(2) * self._speckle_limits
                    speckles.append(new_speckle)
                    raw_grid += np.exp(
                        -(
                            (self.X - new_speckle[0]) ** 2
                            + (self.Y - new_speckle[1]) ** 2
                        )
                        / ((self._speckle_size**2) / 4)
                    )
                    final_grid = np.clip(raw_grid * 200, 0, 255)
                    mi = np.mean(final_grid)
                    i += 1
                    bar(min(mi / self._tmi, 1))
                    print(i)

        return np.asarray(speckles)

    def _mult(self):
        self._pm = np.zeros((self._image_no, np.shape(self._comp)[0]))
        self.noisem = np.zeros((self._image_no, 2))
        if self._mtyp == 0:
            self._mult = np.linspace(
                self._mmin, self._mmax, self._image_no, endpoint=True
            )
        elif self._mtyp == 1:
            self._mult = np.zeros((self._image_no))
            self._mult[1:] = np.logspace(
                np.log10(self._mmin),
                np.log10(self._mmax),
                self._image_no - 1,
                endpoint=True,
            )
        for i in range(self._image_no):
            self._pm[i] = self._comp * self._mult[i]
            self.noisem[i] = self._noise[0] * self._noise[1]  # * self._mult[i]

    def _image_generation(self):
        with alive_bar(
            self._image_no, dual_line=True, bar="blocks", title="Generating images..."
        ) as bar:
            for i in range(self._image_no):
                warp = self._warp(i, self._ref_speckle)
                warp[:, :2] += self._ref_speckle
                # _tar_speckle = (
                #     self._warp(i, self._ref_speckle)[:, :2] + self._ref_speckle
                # )
                # if self._wrap:
                #     _tar_speckle[:, 0] %= self._image_size_x
                #     _tar_speckle[:, 1] %= self._image_size_y
                # _grid = self._grid(i, _tar_speckle)
                _grid = self._grid(i, warp)
                self._create(i, _grid)
                bar()

    def _grid(self, i, warp):
        grid = np.zeros((self._image_size_y, self._image_size_x))
        if self.noisem[i, 0] != 0.0:
            warp[:, :2] = np.random.normal(loc=warp[:, :2], scale=self.noisem[i, 0])
        # covs = np.reshape(warp[:,2:6], (-1,2,2))*self._speckle_size**2
        for j in range(len(warp)):
            a = max(int(warp[j, 1]) - 100, 0)
            b = min(int(warp[j, 1]) + 101, self._image_size_y)
            c = max(int(warp[j, 0]) - 100, 0)
            d = min(int(warp[j, 0]) + 101, self._image_size_x)
            di = np.exp(
                -(
                    (
                        self.X[
                            a:b,
                            c:d,
                        ]
                        - warp[j, 0]
                    )
                    ** 2
                    + (
                        self.Y[
                            a:b,
                            c:d,
                        ]
                        - warp[j, 1]
                    )
                    ** 2
                )
                / ((self._speckle_size**2) / 4)
            )
            grid[
                a:b,
                c:d,
            ] += di
        if self.noisem[i, 1] != 0.0:
            grid = np.random.normal(loc=grid, scale=self.noisem[i, 1])
        grid = np.clip(grid * 200, 0, 255)
        return grid

    def _create(self, i, grid):
        """Method to store generated speckle pattern intensity matrix as a image file.

        Parameters
        ----------

        """
        im = Im.fromarray(np.uint8(grid))
        im.save(self._image_dir + self._name + "_" + str(i) + self._file_format)
        im.close()


class SpeckleResults(SpeckleBase):
    """

    SpeckleResults class for geopyv.

    Parameters
    ----------
    data : dict
        geopyv data dict from Speckle object.


    Attributes
    ----------
    data : dict
        geopyv data dict from Speckle object.

    """

    def __init__(self, data):
        """Initialisation of geopyv SpeckleResults class."""
        self.data = data
