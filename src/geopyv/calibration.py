"""

Calibration module for geopyv.

"""

import logging
import numpy as np
import geopyv as gp
from geopyv.object import Object
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import glob
import re
from scipy.optimize import root
from alive_progress import alive_bar
import traceback

log = logging.getLogger(__name__)


class CalibrationBase(Object):
    """

    Calibration base class to be used as a mixin. Contains plot functionality.

    """

    def __init__(self):
        super().__init__(object_type="Calibration")
        """

        Calibration base class initialiser.

        """

    def inspect(self, *, image_index=0):
        fig, ax = gp.plots.inspect_calibration(data=self.data, image_index=image_index)

    def visualise(
        self,
        *,
        show=True,
        block=True,
        save=None,
    ):
        # Check if solved.
        if self.data["solved"] is not True:
            log.error(
                "Calibration not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.calibration.Calibration.solve()` to solve."
            )
            raise ValueError(
                "Calibration not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.calibration.Calibration.solve()` to solve."
            )
        self._report(gp.check._check_type(show, "show", [bool]), "TypeError")
        self._report(gp.check._check_type(block, "block", [bool]), "TypeError")
        self._report(gp.check._check_type(save, "save", [str, type(None)]), "TypeError")

        gp.plots.visualise_calibration(self.data, block, show, save)

    def contour(
        self,
        *,
        quantity="R",
        points=True,
        colorbar=True,
        ticks=None,
        alpha=0.75,
        levels=None,
        axis=True,
        xlim=None,
        ylim=None,
        show=True,
        block=True,
        save=None,
    ):
        # Check if solved.
        if self.data["solved"] is not True:
            log.error(
                "Calibration not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.calibration.Calibration.solve()` to solve."
            )
            raise ValueError(
                "Calibration not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.calibration.Calibration.solve()` to solve."
            )
        types = [
            "u",
            "v",
            "R",
        ]
        if quantity:
            self._report(
                gp.check._check_value(quantity, "quantity", types), "ValueError"
            )
        self._report(gp.check._check_type(colorbar, "colorbar", [bool]), "TypeError")
        types = [tuple, list, np.ndarray, type(None)]
        self._report(gp.check._check_type(ticks, "ticks", types), "TypeError")
        check = gp.check._check_type(alpha, "alpha", [float])
        if check:
            try:
                alpha = float(alpha)
                self._report(gp.check._conversion(alpha, "alpha", float), "Warning")
            except Exception:
                self._report(check, "TypeError")
        self._report(gp.check._check_range(alpha, "alpha", 0.0, 1.0), "ValueError")
        self._report(gp.check._check_type(levels, "levels", types), "TypeError")
        self._report(gp.check._check_type(axis, "axis", [bool]), "TypeError")
        types = [tuple, list, np.ndarray, type(None)]
        self._report(gp.check._check_type(xlim, "xlim", types), "TypeError")
        if xlim is not None:
            self._report(gp.check._check_dim(xlim, "xlim", 1), "ValueError")
            self._report(gp.check._check_axis(xlim, "xlim", 0, [2]), "ValueError")
        self._report(gp.check._check_type(ylim, "ylim", types), "TypeError")
        if ylim is not None:
            self._report(gp.check._check_dim(ylim, "ylim", 1), "ValueError")
            self._report(gp.check._check_axis(ylim, "ylim", 0, [2]), "ValueError")
        self._report(gp.check._check_type(show, "show", [bool]), "TypeError")
        self._report(gp.check._check_type(block, "block", [bool]), "TypeError")
        self._report(gp.check._check_type(save, "save", [str, type(None)]), "TypeError")

        gp.plots.contour_calibration(
            self.data,
            quantity,
            points,
            colorbar,
            ticks,
            alpha,
            levels,
            axis,
            xlim,
            ylim,
            block,
            show,
            save,
        )

    def error(
        self,
        *,
        quantity="R",
        points=True,
        colorbar=True,
        ticks=None,
        alpha=0.75,
        levels=None,
        axis=True,
        xlim=None,
        ylim=None,
        show=True,
        block=True,
        save=None,
    ):
        # Check if solved.
        if self.data["solved"] is not True:
            log.error(
                "Calibration not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.calibration.Calibration.solve()` to solve."
            )
            raise ValueError(
                "Calibration not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.calibration.Calibration.solve()` to solve."
            )
        types = [
            "u",
            "v",
            "R",
        ]
        if quantity:
            self._report(
                gp.check._check_value(quantity, "quantity", types), "ValueError"
            )
        self._report(gp.check._check_type(points, "points", [bool]), "TypeError")
        self._report(gp.check._check_type(colorbar, "colorbar", [bool]), "TypeError")
        types = [tuple, list, np.ndarray, type(None)]
        self._report(gp.check._check_type(ticks, "ticks", types), "TypeError")
        check = gp.check._check_type(alpha, "alpha", [float])
        if check:
            try:
                alpha = float(alpha)
                self._report(gp.check._conversion(alpha, "alpha", float), "Warning")
            except Exception:
                self._report(check, "TypeError")
        self._report(gp.check._check_range(alpha, "alpha", 0.0, 1.0), "ValueError")
        self._report(gp.check._check_type(levels, "levels", types), "TypeError")
        self._report(gp.check._check_type(axis, "axis", [bool]), "TypeError")
        types = [tuple, list, np.ndarray, type(None)]
        self._report(gp.check._check_type(xlim, "xlim", types), "TypeError")
        if xlim is not None:
            self._report(gp.check._check_dim(xlim, "xlim", 1), "ValueError")
            self._report(gp.check._check_axis(xlim, "xlim", 0, [2]), "ValueError")
        self._report(gp.check._check_type(ylim, "ylim", types), "TypeError")
        if ylim is not None:
            self._report(gp.check._check_dim(ylim, "ylim", 1), "ValueError")
            self._report(gp.check._check_axis(ylim, "ylim", 0, [2]), "ValueError")
        self._report(gp.check._check_type(show, "show", [bool]), "TypeError")
        self._report(gp.check._check_type(block, "block", [bool]), "TypeError")
        self._report(gp.check._check_type(save, "save", [str, type(None)]), "TypeError")

        gp.plots.error_calibration(
            self.data,
            quantity,
            points,
            colorbar,
            ticks,
            alpha,
            levels,
            axis,
            xlim,
            ylim,
            block,
            show,
            save,
        )

    def o2i(self, objpnts):
        """Method for mapping coordinates from object space to image space."""

        # Checks for format of objpnts (should be [x,y,0,1]).
        if np.shape(objpnts)[-1] == 2:
            objpnts = np.pad(objpnts, (0, 1))[:-1]
        if np.shape(objpnts)[-1] == 3:
            objpnts = np.pad(objpnts, (0, 1), constant_values=(0, 1))[:-1]

        # Mapping.
        X_c = self._extmat @ objpnts.T
        X_c /= X_c[2]
        r2 = X_c[0] ** 2 + X_c[1] ** 2
        f = 1 + self._dist[0] * r2 + self._dist[1] * r2**2 + self._dist[4] * r2**3
        X_pp = np.ones((np.shape(objpnts)[0], 3))
        X_pp[:, 0] = (
            X_c[0] * f
            + 2 * self._dist[2] * X_c[0] * X_c[1]
            + self._dist[3] * (r2 + 2 * X_c[0] ** 2)
        )
        X_pp[:, 1] = (
            X_c[1] * f
            + self._dist[2] * (r2 + 2 * X_c[1] ** 2)
            + 2 * self._dist[3] * X_c[0] * X_c[1]
        )

        return (self._intmat @ X_pp.T).T[:, :2]

    def i2o(self, *, imgpnts):
        # Checks for format of imgpnts as in o2i.
        if np.ndim(imgpnts) <= 1:
            imgpnts = imgpnts[np.newaxis, ...]
        if np.shape(imgpnts)[-1] == 2:
            imgpnts = np.pad(imgpnts, (0, 1), constant_values=(0, 1))[:-1]
        if np.shape(imgpnts)[0] >= 30:
            ind = [i for i in range(0, np.shape(imgpnts)[0], 30)]
            ind.append(np.shape(imgpnts)[0])
        else:
            ind = [0, np.shape(imgpnts)[0]]

        objpnts = np.empty((np.shape(imgpnts)[0], 3))
        for i in range(len(ind) - 1):
            X_pp = np.linalg.inv(self._intmat) @ imgpnts[ind[i] : ind[i + 1]].T
            X_c = np.pad(self._simult(X_pp), (0, 1), constant_values=(0, 1))[:, :-1]
            X_c = self._depth_recovery(X_c)
            X_c = np.pad(X_c, (0, 1), constant_values=(0, 1))[:, :-1]
            objpnts[ind[i] : ind[i + 1]] = (np.linalg.inv(self._extmat) @ X_c).T[:, :3]

        return objpnts

    def _depth_recovery(self, X_c):
        invextmat = np.linalg.inv(self._extmat)
        X_c *= -invextmat[2, 3] / (
            invextmat[2, 0] * X_c[0] + invextmat[2, 1] * X_c[1] + invextmat[2, 2]
        )

        return X_c

    def _simult(self, X_pp):
        solution = root(
            lambda X_c: self._equations(X_c, X_pp), np.zeros((2, np.shape(X_pp)[-1]))
        )

        return solution.x.reshape(2, -1)

    def _equations(self, X_c, X_pp):
        X_c = np.reshape(X_c, (2, -1))
        r2 = np.sum(np.square(X_c), axis=0)  # X_c[0] ** 2 + X_c[1] ** 2
        f = 1 + self._dist[0] * r2 + self._dist[1] * r2**2 + self._dist[4] * r2**3
        c = (
            X_c * f
            + 2 * self._dist[2:4, None] * (X_c[0] * X_c[1])
            + self._dist[3:1:-1, None] * (r2 + 2 * X_c**2)
            - X_pp[:2]
        )
        return c.flatten()

    def calibrate(self, *, verbose=True, object=None):
        if self.data["solved"] is not True:
            log.error(
                "Calibration not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.calibration.Calibration.solve()` to solve."
            )
            raise ValueError(
                "Calibration not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.calibration.Calibration.solve()` to solve."
            )
        self._report(
            gp.check._check_type(
                object,
                "object",
                [
                    gp.sequence.Sequence,
                    gp.sequence.SequenceResults,
                    gp.mesh.Mesh,
                    gp.mesh.MeshResults,
                    gp.subset.Subset,
                    gp.subset.SubsetResults,
                    gp.geometry.region.Path,
                    gp.geometry.region.Circle,
                    gp.geometry.region.Region,
                    gp.geometry.region.RegionResults,
                    dict,
                ],
            ),
            "TypeError",
        )

        if type(object) == dict:
            data = object
        else:
            data = object.data

        # Check if already calibrated.
        try:
            if data["calibrated"] is True:
                return
        except Exception:
            pass

        if data["type"] == "Subset":
            if verbose:
                log.info("Calibrating Subset ...")
            # Calibrating.
            X = self.i2o(
                imgpnts=np.asarray(
                    [
                        [
                            data["position"]["x"],
                            data["position"]["y"],
                        ],
                        [
                            data["position"]["x"] + data["results"]["u"],
                            data["position"]["y"] + data["results"]["v"],
                        ],
                    ]
                ),
            )
            # Storing.
            data["position"].update({"X": X[0, 0], "Y": X[0, 1]})
            data["results"].update(
                {
                    "U": X[1, 0] - X[0, 0],
                    "V": X[1, 1] - X[0, 1],
                }
            )
            data["calibrated"] = True
            # Assigning.
            if type(object) == gp.subset.Subset:
                object._X = X[0, 0]
                object._Y = X[0, 1]
                object._U = X[1, 0] - X[0, 0]
                object._V = X[1, 1] - X[0, 1]
                object._calibrated = True
        elif data["type"] == "Mesh":
            if verbose:
                log.info("Calibrating Mesh ...")
            # Calibrating.
            Nodes = self.i2o(
                imgpnts=data["nodes"],
            )[:, :2]
            Centroids = np.mean(Nodes[data["elements"]], axis=-2)
            Displacements = (
                self.i2o(imgpnts=(data["nodes"] + data["results"]["displacements"]))[
                    :, :2
                ]
                - Nodes
            )
            # Distribute.
            for i in range(len(data["results"]["subsets"])):
                s = data["results"]["subsets"][i]
                s["position"].update(
                    {
                        "X": Nodes[i, 0],
                        "Y": Nodes[i, 1],
                    }
                )
                s["results"].update(
                    {
                        "U": Displacements[i, 0],
                        "V": Displacements[i, 1],
                    }
                )
                s["calibrated"] = True

            # Storing.
            data.update(
                {
                    "calibrated": True,
                    "Nodes": Nodes,
                    "Centroids": Centroids,
                }
            )
            data["results"].update(
                {
                    "Displacements": Displacements,
                }
            )
            # Assigning.
            if type(object) != dict:
                object._Nodes = Nodes
                object._Centroids = Centroids
                object._Displacements = Displacements
                object._element_area(space="O")
                object._element_strains(space="O")
                data["results"]["Warps"] = object._Warps
                data["Areas"] = object._Areas
                object._calibrated = True
            if verbose:
                log.info("... Mesh Calibrated.")
        elif data["type"] == "Sequence":
            log.info("Calibrating Sequence ...")
            with alive_bar(
                len(data["meshes"]), bar="blocks", title="Calibrating Meshes ..."
            ) as bar:
                # Recursive.
                if data["file_settings"]["save_by_reference"] is True:
                    for i in range(len(data["meshes"])):
                        mesh = object._load_mesh(i, obj=True, verbose=False)
                        self.calibrate(object=mesh, verbose=False)
                        object._save_mesh(mesh, i, verbose=False)
                        bar()
                else:
                    for i in range(len(data["meshes"])):
                        self.calibrate(object=data["meshes"][i], verbose=False)
                        bar()
            with alive_bar(
                len(data["mesh_settings"]["exclusion_objs"]) + 1,
                bar="blocks",
                title="Calibrating Regions ...",
            ) as bar:
                self.calibrate(object=data["mesh_settings"]["boundary_obj"])
                bar()
                for i in range(len(data["mesh_settings"]["exclusion_objs"])):
                    self.calibrate(object=data["mesh_settings"]["exclusion_objs"][i])
                    bar()

            # Storing.
            data["calibrated"] = True
            # Assigning.
            if type(object) == gp.sequence.Sequence:
                object._calibrated = True
        elif data["type"] == "geometry.Region":
            if verbose:
                log.info("Calibrating Region...")
            # Calibrating.
            Centres = self.i2o(imgpnts=data["centres"])
            shape = np.shape(np.asarray(data["nodes"]))
            Nodes = self.i2o(imgpnts=np.asarray(data["nodes"]).reshape(-1, 2))
            # Storing.
            data.update(
                {
                    "Centres": Centres[:, :2],
                    "Nodes": Nodes[:, :2].reshape(shape),
                }
            )

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


class Calibration(CalibrationBase):
    def __init__(
        self,
        *,
        calibration_dir=".",
        common_name="",
        calibration_file_format=".jpg",
        dictionary=None,
        board_parameters=None,
        show=False,
        save=False,
    ):
        """
        Initialisation of geopyv calibration object.

        Parameters
        ----------
        calibration_dir : str, optional
            Directory of calibration images. Defaults to current working directory.
        common_name : str, optional
            If multiple image sets are within the folder, specifying the common name
            is necessary to identify which set of images to use.
        calibration_file_format : str, optional
                Image file type. Options are ".jpg", ".png" or ".bmp". Defaults to .jpg.

        """

        # Set initialised boolean.
        self.initialised = False

        # Check types.
        self._report(
            gp.check._check_type(calibration_dir, "calibration_dir", [str]), "TypeError"
        )
        if self._report(
            gp.check._check_path(calibration_dir, "calibration_dir"), "Warning"
        ):
            calibration_dir = gp.io._get_image_dir()
        calibration_dir = gp.check._check_character(calibration_dir, "/", -1)
        self._report(
            gp.check._check_type(
                calibration_file_format, "calibration_file_format", [str]
            ),
            "TypeError",
        )
        file_format = gp.check._check_character(calibration_file_format, ".", 0)
        self._report(
            gp.check._check_value(
                calibration_file_format,
                "calibration_file_format",
                [".jpg", ".png", ".bmp", ".tiff"],
            ),
            "ValueError",
        )

        # Store variables.
        self._calibration_dir = calibration_dir
        self._common_name = common_name
        self._file_format = file_format
        self._dictionary = dictionary
        self.solved = False
        self._unsolvable = False

        # Board Setup.
        board_settings = self._board_setup(board_parameters, show, save)

        # Load calibration images.
        try:
            self._calibration_images = glob.glob(
                self._common_name + "*" + self._file_format,
                root_dir=self._calibration_dir,
            )
            image_no = np.shape(self._calibration_images)[0]

            if image_no < 1:
                msg = "No images found in specified folder."
                log.error(msg)
                raise FileExistsError(msg)
            elif image_no < 10:
                msg = (
                    "{} images found. A minimum of 10 images is recommended."
                ).format(image_no)
                log.warning(msg)
            else:
                msg = ("{} images found.").format(image_no)
                log.info(msg)
        except Exception:
            log.error(
                ("Cannot find files with specified path: {}").format(
                    self._calibration_dir + self._common_name + "*" + self._file_format
                )
            )
            print(traceback.format_exc())
            return

        image = cv2.imread(
            self._calibration_dir + self._calibration_images[0], cv2.IMREAD_COLOR
        )
        self._image_size = np.shape(image)
        del image

        # Data.
        file_settings = {
            "calibration_dir": self._calibration_dir,
            "common_name": self._common_name,
            "calibration_images": self._calibration_images,
            "file_format": self._file_format,
            "image_size": self._image_size,
        }
        self.data = {
            "type": "Calibration",
            "solved": self.solved,
            "unsolvable": self._unsolvable,
            "file_settings": file_settings,
            "board_settings": board_settings,
        }

        self._initialised = True

    def _board_setup(self, board_parameters, show, save):
        """Private method for setting up the board."""

        # Board setup.
        columns, rows, square_length, marker_length = board_parameters
        self._board = aruco.CharucoBoard(
            (columns, rows), square_length, marker_length, self._dictionary
        )
        self._objpnts = self._board.getChessboardCorners()
        self._board_corners = self._board.getChessboardCorners()
        board_settings = {
            # "dictionary": self._dictionary,
            "columns": columns,
            "rows": rows,
            "corners": self._board_corners,
            "objpnts": self._objpnts,
            "square_length": square_length,
            "marker_length": marker_length,
        }

        if show:
            imboard = self._board.generateImage((290, 180))
            if save:
                cv2.imwrite(save + ".tiff", imboard)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.imshow(imboard, interpolation="nearest")
            ax.axis("off")
            plt.show()

        return board_settings

    def solve(self, *, index=None, binary=None, threshold=50):
        """
        Private method to calibrate the camera.
        """

        # Input checks.
        check = gp.check._check_type(binary, "binary", [float, int, type(None)])
        if check:
            try:
                binary = float(binary)
                self._report(gp.check._conversion(binary, "binary", float), "Warning")
            except Exception:
                self._report(check, "TypeError")
        self._report(
            gp.check._check_type(threshold, "threshold", [int, float]), "TypeError"
        )
        check = self._report(gp.check._check_type(index, "index", [int]), "Warning")
        if check:
            _ext_img = gp.io._load_img("extrinsic", load=False)
            index = int(re.findall(r"\d+", _ext_img)[-1])
        image_indices = np.asarray(
            [int(re.findall(r"\d+", x)[-1]) for x in self._calibration_images]
        )
        try:
            index = np.argwhere(image_indices == index)[0][0]
        except Exception:
            _ext_img = gp.io._load_img("extrinsic", load=False)
            index = int(re.findall(r"\d+", _ext_img)[-1])
            index = np.argwhere(image_indices == index)[0][0]

        # Store input.
        self._binary = binary
        self._index = index
        self._img_index = index + 1

        # Pre-process images for calibration.
        self._allCorners, self._allIds, self._imsize = self._read_chessboards(threshold)
        if len(self._allCorners) == 0:
            log.error("Unsolvable: no corners identified.")
            return
        self._imgpnts = []
        for i in range(len(self._allCorners)):
            self._imgpnts.append(self._allCorners[i].reshape(-1, 2))

        # Calibrate camera.
        self._calibrate_camera(self._allCorners, self._allIds, self._imsize)

        # Post-processing.
        self._extrinsic_matrix_generator()
        self._camera_angles()
        self._reprojection()

        # Store data.
        self.solved = True
        self.data.update(
            {
                "solved": self.solved,
                "intrinsic_matrix": self._intmat,
                "extrinsic_matrix": self._extmat,
                "distortion": self._dist,
                "camera_angles": [
                    self._alpha,
                    self._beta,
                    self._gamma,
                ],
                "projection": {
                    "imgpnts": self._imgpnts,
                    "objpnts": self._objpnts,
                    "reimgpnts": self._reimgpnts,
                },
                "calibration": {
                    "ret": self._ret,
                    "rotation": self._rot,
                    "translation": self._trans,
                    "corners": self._allCorners,
                    "ids": self._allIds,
                },
            }
        )

    def _read_chessboards(self, threshold):
        allCorners = []
        allIds = []
        acceptedImages = []

        # Sub-pixel detection criteria.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        with alive_bar(
            np.shape(self._calibration_images)[0],
            dual_line=True,
            bar="blocks",
            title="Extracting...",
        ) as bar:
            for i in range(len(self._calibration_images)):
                bar.text = "Processing image {}...".format(
                    self._calibration_dir + self._calibration_images[i]
                )

                # Image loading.
                frame = cv2.imread(self._calibration_dir + self._calibration_images[i])
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.GaussianBlur(frame, (5, 5), 0)

                # Image binarisation (optional).
                if self._binary is not None:
                    ret, bin_frame = cv2.threshold(
                        frame, self._binary, 255, cv2.THRESH_BINARY
                    )

                    # Marker Detection.
                    arcnrs, arids, rejectedImgPoints = cv2.aruco.detectMarkers(
                        bin_frame,
                        self._dictionary,
                        parameters=cv2.aruco.DetectorParameters(),
                    )
                else:
                    # Marker Detection.
                    arcnrs, arids, rejectedImgPoints = cv2.aruco.detectMarkers(
                        frame,
                        self._dictionary,
                        parameters=cv2.aruco.DetectorParameters(),
                    )

                if len(arcnrs) > 0:
                    for arcnr in arcnrs:
                        # Sub-pixel refinement.
                        cv2.cornerSubPix(
                            frame,
                            arcnr,
                            winSize=(3, 3),
                            zeroZone=(-1, -1),
                            criteria=criteria,
                        )
                    # ChArUco interpolation.
                    ret, chcnrs, chids = cv2.aruco.interpolateCornersCharuco(
                        arcnrs, arids, frame, self._board
                    )
                    if chcnrs is not None and chids is not None and len(chcnrs) > 3:
                        if len(chcnrs) > threshold:
                            allCorners.append(chcnrs)
                            allIds.append(chids)
                            acceptedImages.append(self._calibration_images[i])
                bar()
        self._calibration_images = acceptedImages
        image_indices = np.asarray(
            [int(re.findall(r"\d+", x)[-1]) for x in self._calibration_images]
        )
        try:
            self._index = np.argwhere(image_indices == self._img_index)[0][0]
        except Exception:
            log.error("External matrix image of insufficient quality.")
        imsize = frame.shape
        return allCorners, allIds, imsize

    def _calibrate_camera(self, allCorners, allIds, imsize):
        """Private method to calibrate the camera."""

        log.info("Calibrating camera...")
        cameraMatrixInit = np.array(
            [
                [1000.0, 0.0, imsize[0] / 2.0],
                [0.0, 1000.0, imsize[1] / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        distCoeffsInit = np.zeros((5, 1))
        (
            self._ret,
            self._intmat,
            _dist,
            _rot,
            _trans,
        ) = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=allCorners,
            charucoIds=allIds,
            board=self._board,
            imageSize=imsize,
            cameraMatrix=cameraMatrixInit,
            distCoeffs=distCoeffsInit,
        )

        self._trans = np.asarray(_trans)
        self._rot = np.asarray(_rot)
        self._dist = _dist.flatten()

    def _reprojection(self):
        """Private method to generate reproject from object space
        to image space for the calibrationcoordinates for the purpose
        of error estimation."""

        self._reimgpnts = []
        for index in range(len(self._allCorners)):
            objpnts = self._find_objpnts(index)
            X_c = self._extmats[index] @ objpnts.T
            X_c /= X_c[2]
            r2 = X_c[0] ** 2 + X_c[1] ** 2
            f = (
                1
                + self._dist[0] * r2
                + self._dist[1] * r2**2
                + self._dist[4] * r2**3
            )
            X_pp = np.ones((np.shape(objpnts)[0], 3))
            X_pp[:, 0] = (
                X_c[0] * f
                + 2 * self._dist[2] * r2 * X_c[0] * X_c[1]
                + self._dist[3] * (r2 + 2 * X_c[0] ** 2)
            )
            X_pp[:, 1] = (
                X_c[1] * f
                + self._dist[2] * (r2 + 2 * X_c[1] ** 2)
                + 2 * self._dist[3] * r2 * X_c[0] * X_c[1]
            )
            self._reimgpnts.append(((self._intmat @ X_pp.T).T)[:, :2])

    def _find_objpnts(self, index):
        objpnts = np.ones((len(self._allIds[index]), 4))
        objpnts[:, :3] = self._objpnts[self._allIds[index]].flatten().reshape(-1, 3)
        return objpnts

    def _extrinsic_matrix_generator(self):
        """Private method for constructing the extrinsic matrices for calibration
        and the extrinsic matrix for mapping."""

        self._extmats = np.zeros((np.shape(self._rot)[0], 4, 4))
        self._extmats[:, :3, 3] = self._trans[:, :3].reshape(-1, 3)
        self._extmats[:, 3, 3] = 1
        for i in range(np.shape(self._extmats)[0]):
            self._extmats[i, :3, :3], _ = cv2.Rodrigues(self._rot[i])
        self._extmat = self._extmats[self._index]

    def _camera_angles(self):
        self._beta = -np.arcsin(self._extmat[2, 0])
        self._gamma = np.arcsin(self._extmat[2, 1] / np.cos(self._beta))
        self._alpha = np.arcsin(self._extmat[1, 0] / np.cos(self._beta))


class CalibrationResults(CalibrationBase):
    """
    Calibration Results class for geopyv.

    Parameters
    ----------
    data : dict
        geopyv data dict from Calibration object.


    Attributes
    ----------
    data : dict
        geopyv data dict from Calibration object.

    """

    def __init__(self, data):
        """Initialisation of geopyv CalibrationResults class."""
        self.data = data
        self._intmat = self.data["intrinsic_matrix"]
        self._extmat = self.data["extrinsic_matrix"]
        self._dist = self.data["distortion"]
