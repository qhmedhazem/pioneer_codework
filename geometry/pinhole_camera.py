from functools import lru_cache
import torch
import torch.nn as nn

from geometry.pose import Pose
from lib.camera_helpers import scale_intrinsics, load_intrinsics, construct_from_array
from lib.image_helpers import image_grid
import cv2
import numpy as np
import glob

########################################################################################################################


class PinholeCamera(nn.Module):
    """
    Differentiable camera class implementing reconstruction and projection
    functions for a pinhole model.
    """

    def __init__(self, K=None, D=None, res=None, Tcw=None):
        """
        Initializes the Camera class

        Parameters
        ----------
        K : torch.Tensor [3,3]
            Camera intrinsics'
        D : torch.Tensor [B,1,5]
            Distortion coefficients
        res : tuple (int, int)
            Resolution of the camera (height, width)
        Tcw : Pose
            Camera -> World pose transformation
        """
        super().__init__()
        self.K = K
        self.Kinv = np.linalg.inv(K)
        self.D = D
        self.width = res[1]
        self.height = res[0]

        if Tcw is not None:
            self.Tcw = Tcw
        elif self.K is not None:
            self.Tcw = Pose.identity(len(self.K))

    @staticmethod
    def from_calib_file(path):
        K, D, res = load_intrinsics(path)

        # Convert K and D to 1D arrays if they are not already
        K = np.array(K)
        D = np.array(D).reshape(-1)
        return PinholeCamera(
            K,
            D,
            res,
        )

    def __len__(self):
        """Batch size of the camera intrinsics"""
        return len(self.K)

    def to(self, *args, **kwargs):
        """Moves object to a specific device"""
        self.K = self.K.to(*args, **kwargs)
        self.Tcw = self.Tcw.to(*args, **kwargs)
        return self

    ########################################################################################################################

    @property
    def fx(self):
        """Focal length in x"""
        return self.K[0, 0]

    @property
    def fy(self):
        """Focal length in y"""
        return self.K[1, 1]

    @property
    def cx(self):
        """Principal point in x"""
        return self.K[0, 2]

    @property
    def cy(self):
        """Principal point in y"""
        return self.K[1, 2]

    @property
    def k1(self):
        """Radial distortion coefficient k1"""
        return self.D[0]

    @property
    def k2(self):
        """Radial distortion coefficient k2"""
        return self.D[1]

    @property
    def p1(self):
        """Tangential distortion coefficient p1"""
        return self.D[2]

    @property
    def p2(self):
        """Tangential distortion coefficient p2"""
        return self.D[3]

    @property
    def xi(self):
        """Affine distortion coefficient xi"""
        """This is not a standard pinhole camera parameter, but included for completeness."""
        return self.D[4]

    @property
    def row(self):
        """Height of the camera image"""
        return self.height

    @property
    def col(self):
        """Width of the camera image"""
        return self.width

    # @property
    # @lru_cache()
    # def Twc(self):
    #     """World -> Camera pose transformation (inverse of Tcw)"""
    #     return self.Tcw.inverse()

    # @property
    # def Kinv(self):
    #     """Inverse intrinsics (for lifting)"""
    #     Kinv = self.K.clone()
    #     Kinv[:, 0, 0] = 1.0 / self.fx
    #     Kinv[:, 1, 1] = 1.0 / self.fy
    #     Kinv[:, 0, 2] = -1.0 * self.cx / self.fx
    #     Kinv[:, 1, 2] = -1.0 * self.cy / self.fy
    #     return Kinv

    def scaled(self, x_scale, y_scale=None):
        """
        Returns a scaled version of the camera (changing intrinsics)

        Parameters
        ----------
        x_scale : float
            Resize scale in x
        y_scale : float
            Resize scale in y. If None, use the same as x_scale

        Returns
        -------
        camera : Camera
            Scaled version of the current cmaera
        """
        # If single value is provided, use for both dimensions
        if y_scale is None:
            y_scale = x_scale
        # If no scaling is necessary, return same camera
        if x_scale == 1.0 and y_scale == 1.0:
            return self
        # Scale intrinsics and return new camera with same Pose
        K = scale_intrinsics(self.K.clone(), x_scale, y_scale)
        return PinholeCamera(K, Tcw=self.Tcw)

        ########################################################################################################################

    def calibrate(
        self, images, pattern_size=(18, 25), square_size=1.0, device=None, headless=True
    ):
        """
        Recalibrates the camera intrinsics using a set of images of a calibration pattern.

        Parameters
        ----------
        images : list of np.ndarray
            List of grayscale images (as numpy arrays) to use for calibration.
        pattern_size : tuple (int, int)
            Number of inner corners per chessboard row and column.
        square_size : float
            Size of a square in your defined unit (point, millimeter, etc.).
        device : torch.device, optional
            Device to move the new intrinsics to.

        Returns
        -------
        Camera
            New camera instance with recalibrated intrinsics.
        """
        if not images or not isinstance(images, list):
            raise ValueError("A non-empty list of images must be provided.")

        # Avoiding conflect because pattern size
        pattern_size = (pattern_size[0] - 1, pattern_size[1] - 1)

        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(
            -1, 2
        )
        objp *= square_size

        objpoints = []
        imgpoints = []

        for img in images:
            if img is None or len(img.shape) != 2:
                continue
            ret, corners = cv2.findChessboardCorners(img, pattern_size, None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners.squeeze())
                if not headless:
                    cv2.drawChessboardCorners(img, pattern_size, corners, ret)
                    cv2.imshow("Chessboard Corners", img)
                    cv2.waitKey(500)

        if len(objpoints) < 3:
            raise RuntimeError("Not enough valid calibration images found.")

        cv2.destroyAllWindows()

        img_shape = images[0].shape[::-1]

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None
        )

        self.K = (
            torch.tensor(K, dtype=self.K.dtype, device=device or self.K.device)
            .unsqueeze(0)
            .repeat(len(self.K), 1, 1)
        )
        return K, dist

    def distortion(self, p_u):
        """
        Apply radial + tangential distortion to normalized points.

        Parameters:
        -----------
        p_u : np.ndarray [N, 2] - undistorted normalized points

        Returns:
        --------
        d_u : np.ndarray [N, 2] - distortion vectors
        """
        k1 = self.k1
        k2 = self.k2
        p1 = self.p1
        p2 = self.p2

        x = p_u[0]
        y = p_u[1]

        mx2_u = x * x
        my2_u = y * y
        mxy_u = x * y
        rho2_u = mx2_u + my2_u
        rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u

        dx = x * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u)
        dy = y * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u)

        return np.stack([dx, dy], axis=-1)

    def lift_projective(self, p):
        """
        Convert pixel point(s) to normalized 3D rays.

        Parameters:
        -----------
        p : np.ndarray [2] or [N, 2]

        Returns:
        --------
        rays : np.ndarray [3] or [N, 3]
        """
        mx_d = self.Kinv[0, 0] * p[0] + self.Kinv[0, 2]
        my_d = self.Kinv[1, 1] * p[1] + self.Kinv[1, 2]
        mx_u = None
        my_u = None

        for _ in range(8):
            p_u = np.stack([mx_d, my_d], axis=-1)
            d_u = self.distortion(p_u)
            mx_u = mx_d - d_u[0]
            my_u = my_d - d_u[1]

        return mx_u, my_u, 1.0

    def undistort(self, img):
        undistorted = cv2.undistort(img, self.K, self.D)
        return undistorted
