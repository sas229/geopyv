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

    def inspect_(self, *, image_index=0):
        plt.figure()
        frame = cv2.imread(self._calibration_images[image_index])
        img_undist = cv2.undistort(frame, self._camera_matrix, self._dist, None)
        plt.subplot(1, 2, 1)
        plt.imshow(frame)

        plt.title("Raw image")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_undist)
        plt.title("Corrected image")
        plt.axis("off")
        plt.show()

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
        calibration_common_name="",
        calibration_file_format=".jpg",
        method="charuco",
        dictionary=None,
        board_parameters=None
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
                [".jpg", ".png", ".bmp"],
            ),
            "ValueError",
        )

        # Store variables.
        self._calibration_dir = calibration_dir
        self._file_format = file_format
        self._method = method
        self._dictionary = dictionary
        self.solved = False
        self._unsolvable = False

        # Board setup.
        if self._method == "charuco":
            columns, rows, square_length, marker_length = board_parameters
            self._board = aruco.CharucoBoard(
                (columns, rows), square_length, marker_length, self._dictionary
            )
            imboard = self._board.generateImage((2000, 2000))
            cv2.imwrite("chessboard.tiff", imboard)
            # fig = plt.figure()
            # ax = fig.add_subplot(1,1,1)
            # plt.imshow(imboard, cmap = mpl.cm.gray, interpolation = "nearest")
            # ax.axis("off")
            # plt.show()

        try:
            self._calibration_images = glob.glob(
                self._calibration_dir + "*" + self._file_format
            )
        except Exception:
            log.error(
                "Issues encountered recognising image file names. "
                "Please refer to the documentation for naming guidance."
            )
        board_settings = {
            "dictionary": self._dictionary,
            "board": self._board,
            "columns": columns,
            "rows": rows,
            "square_length": square_length,
            "marker_length": marker_length,
        }
        file_settings = {
            "calibration_dir": self._calibration_dir,
            "calibration_images": self._calibration_images,
            "file_format": self._file_format,
        }

        self.data = {
            "type": "Calibration",
            "solved": self.solved,
            "unsolvable": self._unsolvable,
            "file_settings": file_settings,
            "method": self._method,
            "board_settings": board_settings,
        }

        self._calibrate()
        self._calibration = {
            "ret": self._ret,
            "camera_matrix": self._camera_matrix,
            "distortion": self._dist,
            "rotation": self._rot,
            "translation": self._trans,
            "corners": self._allCorners,
            "ids": self._allIds,
        }
        self.data.update({"calibration": self._calibration})

        self._initialised = True

    def _calibrate(self):
        """
        Private method to calibrate the camera.
        """
        if self._method == "charuco":
            self._allCorners, self._allIds, self._imsize = self._read_chessboards()
            print(self._allCorners)
            (
                self._ret,
                self._camera_matrix,
                self._dist,
                self._rot,
                self._trans,
            ) = self._calibrate_camera(self._allCorners, self._allIds, self._imsize)

    def _read_chessboards(self):
        allCorners = []
        allIds = []
        decimator = 0
        # SUB PIXEL CORNER DETECTION CRITERION
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        for im in self._calibration_images:
            log.info("Processing image {}...".format(im))
            frame = cv2.imread(im)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
                gray, self._dictionary
            )
            if len(corners) > 0:
                for corner in corners:
                    cv2.cornerSubPix(
                        gray,
                        corner,
                        winSize=(3, 3),
                        zeroZone=(-1, -1),
                        criteria=criteria,
                    )
                res2 = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, self._board
                )
                if (
                    res2[1] is not None
                    and res2[2] is not None
                    and len(res2[1]) > 3
                    and decimator % 1 == 0
                ):
                    allCorners.append(res2[1])
                    allIds.append(res2[2])
                else:
                    allCorners.append(np.asarray([]))
                    allIds.append(np.asarray([]))

            decimator += 1

        imsize = gray.shape
        return allCorners, allIds, imsize

    def _calibrate_camera(self, allCorners, allIds, imsize):
        log.info("Calibrating camera...")
        cameraMatrixInit = np.array(
            [
                [1000.0, 0.0, imsize[0] / 2.0],
                [0.0, 1000.0, imsize[1] / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )

        distCoeffsInit = np.zeros((5, 1))
        flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS
            + cv2.CALIB_RATIONAL_MODEL
            + cv2.CALIB_FIX_ASPECT_RATIO
        )

        (
            ret,
            camera_matrix,
            distortion_coefficients0,
            rotation_vectors,
            translation_vectors,
            stdDeviationsIntrinsics,
            stdDeviationsExtrinsics,
            perViewErrors,
        ) = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=allCorners,
            charucoIds=allIds,
            board=self._board,
            imageSize=imsize,
            cameraMatrix=cameraMatrixInit,
            distCoeffs=distCoeffsInit,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9),
        )

        return (
            ret,
            camera_matrix,
            distortion_coefficients0,
            rotation_vectors,
            translation_vectors,
        )

    def solve(
        self, *, image_dir=".", common_name="", file_format=".jpg", output_dir="."
    ):
        """
        Method to solve for the calibration.
        """
        # Check types.
        self._report(gp.check._check_type(image_dir, "image_dir", [str]), "TypeError")
        if self._report(gp.check._check_path(image_dir, "image_dir"), "Warning"):
            image_dir = gp.io._get_image_dir()
        image_dir = gp.check._check_character(image_dir, "/", -1)
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
        if self._report(gp.check._check_path(output_dir, "output_dir"), "Warning"):
            output_dir = gp.io._get_image_dir()
        output_dir = gp.check._check_character(output_dir, "/", -1)

        self._image_dir = image_dir

        self._common_name = common_name
        self._file_format = file_format
        _images = glob.glob(
            self._image_dir + self._common_name + "*" + self._file_format
        )
        _image_indices_unordered = [int(re.findall(r"\d+", x)[-1]) for x in _images]
        _image_indices_arguments = np.argsort(_image_indices_unordered)
        self._images = [_images[index] for index in _image_indices_arguments]
        self._image_indices = np.sort(_image_indices_unordered)

        i = 0
        for image in self._images:
            frame = cv2.imread(image)
            img_undist = cv2.undistort(frame, self._camera_matrix, self._dist, None)
            cv2.imwrite(
                output_dir + "calibrated_" + str(i) + self._file_format, img_undist
            )
            del frame
            i += 1


# def calibrate(
#    image_folder=".",
#    common_name="",
#    file_format=".jpg",)
