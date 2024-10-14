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

    def _warp(self, i, ref):
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
            elif self.data["vars"]["mode"] == "T":
                if i == 0:
                    pass
                else:
                    # Points.
                    C = self.data["vars"]["centre"] + np.asarray(
                        [0.0, self.data["vars"]["rate"] * i]
                    )
                    X = np.asarray(
                        [
                            self.data["vars"]["lambda"]
                            * self.data["vars"]["radius"]
                            / np.cos(np.pi / 2 - self.data["vars"]["b2"]),
                            C[1],
                        ]
                    )
                    m = -1 / np.tan(np.pi / 2 - self.data["vars"]["b2"])
                    coefs = np.asarray(
                        [
                            1 + m**2,
                            2 * (m * (X[0] - C[0]) - X[1]) - 2 * m**2 * C[1],
                            (m * (X[0] - C[0]) - X[1]) ** 2
                            + m**2 * (C[1] ** 2 - self.data["vars"]["radius"] ** 2),
                        ]
                    )
                    roots = np.roots(coefs)
                    B = np.asarray(
                        [
                            C[0]
                            + np.sqrt(
                                self.data["vars"]["radius"] ** 2
                                - (roots[np.argwhere(roots > C[1])][0][0] - C[1]) ** 2
                            ),
                            roots[np.argwhere(roots > C[1])][0][0],
                        ]
                    )
                    Z = np.asarray(
                        [
                            C[0]
                            + self.data["vars"]["lambda"]
                            * self.data["vars"]["radius"]
                            * np.sin(self.data["vars"]["b1"]),
                            C[1]
                            + self.data["vars"]["lambda"]
                            * self.data["vars"]["radius"]
                            * np.cos(self.data["vars"]["b1"]),
                        ]
                    )
                    m = -1 / np.tan(np.pi / 2 - self.data["vars"]["b1"])
                    coefs = np.asarray(
                        [
                            1 + m**2,
                            2 * (m * (Z[0] - C[0]) - Z[1]) - 2 * m**2 * C[1],
                            (m * (Z[0] - C[0]) - Z[1]) ** 2
                            + m**2 * (C[1] ** 2 - self.data["vars"]["radius"] ** 2),
                        ]
                    )
                    roots = np.roots(coefs)
                    D = np.asarray(
                        [
                            C[0]
                            + np.sqrt(
                                self.data["vars"]["radius"] ** 2
                                - (roots[np.argwhere(roots > Z[1])][0][0] - C[1]) ** 2
                            ),
                            roots[np.argwhere(roots > Z[1])][0][0],
                        ]
                    )
                    G = np.asarray([C[0], D[1] - m * (D[0] - C[0])])

                    # Critical Radii.
                    Rb = np.sqrt(np.sum((X - B) ** 2))
                    Rd = np.sqrt(np.sum((X - D) ** 2))
                    Rg = np.sqrt(np.sum((X - G) ** 2))

                    # print("C: {}".format(np.round(C,2)))
                    # print("X: {}".format(np.round(X,2)))
                    # print("m: {}".format(np.round(m,2)))
                    # print("coefs: {}".format(np.round(coefs,2)))
                    # print("roots: {}".format(np.round(roots,2)))
                    # print("B: {}".format(np.round(B,2)))
                    # print("Z: {}".format(np.round(Z,2)))
                    # print("D: {}".format(np.round(D,2)))
                    # print("G: {}".format(np.round(G,2)))
                    # input()

                    # General speckle.
                    Rp = np.sqrt(np.sum((X - ref) ** 2, axis=1))
                    alpha = np.arctan(abs((ref[:, 1] - X[1]) / (ref[:, 0] - X[0])))

                    # Conditions
                    c4 = np.full(np.shape(ref)[0], False)
                    for i in range(len(ref)):
                        if abs(ref[i, 0]) > D[0]:
                            pass
                        else:
                            cond1 = gp.geometry.utilities.intersect(
                                Z, ref[i], D, np.asarray([C[0], D[1]])
                            )
                            cond2 = gp.geometry.utilities.intersect(
                                np.asarray([Z[0], C[1] - (Z[1] - C[1])]),
                                ref[i],
                                np.asarray([D[0], C[1] - (D[1] - C[1])]),
                                np.asarray([C[0], C[1] - (D[1] - C[1])]),
                            )
                            if cond1 or cond2:
                                c4[i] = True

                    c1 = Rp < Rb
                    c2 = (Rp > Rb) * (Rp < Rd)
                    c3 = (Rp > Rd) * (Rp < Rg) * np.logical_not(c4)
                    c4 *= Rp < Rg

                    # Rigid inner.
                    # theta = np.arctan(1/(X[0]-C[0]))
                    # _warp[:,0] = c1 * (np.cos(theta) * (ref[:,0]-X[0])
                    # - np.sin(theta) * (ref[:,1]-X[1])) - (ref[:,0]-X[0])
                    # _warp[:,1] = c1 * (np.cos(theta) * (ref[:,1]-X[1])
                    # - np.sin(theta) * (ref[:,0]-X[0])) - (ref[:,1]-X[1])

                    _warp[:, 0] -= (
                        c1
                        * self.data["vars"]["rate"]
                        * Rp
                        / (X[0] - C[0])
                        * np.sin(alpha)
                        * np.sign(X[1] - ref[:, 1])
                    )
                    _warp[:, 1] -= (
                        c1
                        * self.data["vars"]["rate"]
                        * Rp
                        / (X[0] - C[0])
                        * np.cos(alpha)
                        * np.sign(ref[:, 0] - X[0])
                    )
                    # # Shear zone.
                    _warp[:, 0] -= (
                        c2
                        * self.data["vars"]["rate"]
                        * Rb
                        / (X[0] - C[0])
                        * np.sin(alpha)
                        * np.sign(X[1] - ref[:, 1])
                    )
                    _warp[:, 1] -= (
                        c2
                        * self.data["vars"]["rate"]
                        * Rb
                        / (X[0] - C[0])
                        * np.cos(alpha)
                        * np.sign(ref[:, 0] - X[0])
                    )
                    # # Shear depletion.
                    _warp[:, 0] -= (
                        c3
                        * self.data["vars"]["rate"]
                        * Rp
                        / (X[0] - C[0])
                        * np.sin(alpha)
                        * (Rg - Rp)
                        / (Rg - Rd)
                        * np.sign(X[1] - ref[:, 1])
                    )
                    _warp[:, 1] -= (
                        c3
                        * self.data["vars"]["rate"]
                        * Rp
                        / (X[0] - C[0])
                        * np.cos(alpha)
                        * (Rg - Rp)
                        / (Rg - Rd)
                        * np.sign(ref[:, 0] - X[0])
                    )
                    # # Head.
                    _warp[:, 1] += c4 * self.data["vars"]["rate"]

                    # fig,ax = plt.subplots()
                    # ax.scatter(ref[:,0]*c1, ref[:,1]*c1, color = "r")
                    # ax.scatter(ref[:,0]*c2, ref[:,1]*c2, color = "g")
                    # ax.scatter(ref[:,0]*c3, ref[:,1]*c3, color = "b")
                    # # ax.scatter(ref[:,0]*c4, ref[:,1]*c4, color = "y")
                    # others = np.logical_not(c1)
                    # *np.logical_not(c2)*np.logical_not(c3)#*np.logical_not(c4)
                    # ax.scatter(ref[:,0]*others, ref[:,1]*others, color = "orange")
                    # points = np.asarray([C,X,B,Z,D,G])
                    # ax.scatter(points[:,0],points[:,1],color = "k")
                    # ax.plot((X[0],B[0]), (X[1],B[1]), color = "k")
                    # ax.plot((Z[0],D[0],G[0]), (Z[1],D[1],G[1]), color = "k")
                    # plt.show()
            elif self.data["vars"]["mode"] == "C":
                delta = ref - self.data["vars"]["centre"]
                dist = np.sqrt(np.sum(delta**2, axis=1))
                dtheta = 2 * np.pi * self.data["mult"][i]
                c1 = dist < self.data["vars"]["r1"]
                c2 = np.logical_not(c1) * (dist < self.data["vars"]["r2"])
                m = (self.data["vars"]["r2"] - dist) / (
                    self.data["vars"]["r2"] - self.data["vars"]["r1"]
                )
                _warp[:, 0] += c1 * (
                    self.data["pm"][i][2] * delta[:, 0]
                    + self.data["pm"][i][4] * delta[:, 1]
                )
                _warp[:, 1] += c1 * (
                    self.data["pm"][i][3] * delta[:, 0]
                    + self.data["pm"][i][5] * delta[:, 1]
                )
                if i != 0:
                    for j in range(i):
                        dtheta = 2 * np.pi * (self.data["mult"][j+1] - self.data["mult"][j])/1000
                        for k in range(1000):
                            _warp[:, 2] += c1 * (np.cos(dtheta) - 1)
                            _warp[:, 3] += c1 * (np.sin(dtheta))
                            _warp[:, 4] += c1 * (-np.sin(dtheta))
                            _warp[:, 5] += c1 * (np.cos(dtheta) - 1)
                            _warp[:, 0] += c2 * (
                                np.cos(dtheta * m) * delta[:, 0]
                                - np.sin(dtheta * m) * delta[:, 1]
                                - delta[:, 0]
                            )
                            _warp[:, 1] += c2 * (
                                np.sin(dtheta * m) * delta[:, 0]
                                + np.cos(dtheta * m) * delta[:, 1]
                                - delta[:, 1]
                            )
                            den = np.sqrt(delta[:,0]**2+delta[:,1]**2)*(self.data["vars"]["r2"]-self.data["vars"]["r1"])
                            _warp[:, 2] += c2 * (np.cos(m*dtheta) + dtheta*delta[:,0]/den *(delta[:,0]*np.sin(m*dtheta) + delta[:,1]*np.cos(m*dtheta))-1)
                            _warp[:, 3] += c2 * (np.sin(m*dtheta) + dtheta*delta[:,0]/den * (delta[:,1]*np.sin(m*dtheta) - delta[:,0]*np.cos(m*dtheta)))
                            _warp[:, 4] += c2 * (-np.sin(m*dtheta) + dtheta*delta[:,1]/den * (delta[:,0]*np.sin(m*dtheta) + delta[:,1]*np.cos(m*dtheta)))
                            _warp[:, 5] += c2 * (np.cos(m*dtheta) + dtheta*delta[:,1]/den * (delta[:,1]*np.sin(m*dtheta) - delta[:,0]*np.cos(m*dtheta))-1)

                            # Update 
                            cur_ref = ref + _warp[:,:2]
                            delta = ref + _warp[:,:2] - self.data["vars"]["centre"]
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
        speckle_limits=np.asarray([1001, 1001]),
        image_no=101,
        mmin=0,
        mmax=1,
        mtyp=0,
        rot=False,
        comp=np.zeros(12),
        origin=np.asarray([501.0, 501.0]),
        noise=np.asarray([[0.0, 0.0], [0, 0]]),
        tmi=100,
        speckle_size=10.0,
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
        self._rot = rot
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

    def solve(self, *, wrap=False, number=None, ref_speckle=None, vars=None):
        self._wrap = wrap
        self._vars = vars
        [self.X, self.Y] = np.meshgrid(
            range(self._image_size_x), range(self._image_size_y)
        )
        if ref_speckle is None:
            self._ref_speckle = self._speckle_distribution(number)
        else:
            self._ref_speckle = ref_speckle
        if self._vars is not None:
            if self._vars["mode"] == "T":
                self._speckle = self._ref_speckle
            elif self._vars["mode"] == "C":
                self._rot = True
        self._mult()
        self.data.update(
            {
                "wrap": self._wrap,
                "ref_speckle": self._ref_speckle,
                "pm": self._pm,
                "vars": self._vars,
                "mult": self._mult,
                "rot": self._rot,
                "noisem": self.noisem
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
                    0.5 * (self._image_size_x - self._speckle_limits[0]),
                    0.5 * (self._image_size_y - self._speckle_limits[1]),
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
                            0.5 * self._image_size_y - self._speckle_limits[1],
                            0.5 * self._image_size_x - self._speckle_limits[0],
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
        if self._vars is not None:
            if self._vars["mode"] == "T":
                speckles = speckles[
                    np.argwhere(
                        np.sqrt(np.sum((speckles - self._vars["centre"]) ** 2, axis=1))
                        > self._vars["radius"]
                    ).flatten()
                ]

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
        self.noisem[1:] = self._noise[0] * self._noise[1]
        if self._rot is False:
            for i in range(self._image_no):
                self._pm[i] = self._comp * self._mult[i]
        else:
            for i in range(self._image_no):
                self._pm[i, 2] = np.cos(2 * np.pi * self._mult[i]) - 1
                self._pm[i, 3] = np.sin(2 * np.pi * self._mult[i])
                self._pm[i, 4] = -np.sin(2 * np.pi * self._mult[i])
                self._pm[i, 5] = np.cos(2 * np.pi * self._mult[i]) - 1               

    def _image_generation(self):
        with alive_bar(
            self._image_no, dual_line=True, bar="blocks", title="Generating images..."
        ) as bar:
            for i in range(self._image_no):
                if self._vars is not None:
                    if self._vars["mode"] == "T":
                        warp = self._warp(i, self._speckle)
                        self._speckle += warp[:, :2]
                    else:
                        warp = self._warp(i, self._ref_speckle)
                else:
                    warp = self._warp(i, self._ref_speckle)
                warp[:, :2] += self._ref_speckle
                _grid = self._grid(i, warp)
                self._create(i, _grid)
                bar()

    def _grid(self, i, warp):
        grid = np.zeros((self._image_size_y, self._image_size_x))
        if self.noisem[i, 0] != 0.0:
            warp[:, :2] = np.random.normal(loc=warp[:, :2], scale=self.noisem[i, 0])
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
