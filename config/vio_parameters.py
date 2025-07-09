from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from config.config import Config, _load_yaml, _reshape_4x4


# ---------------------------------------------------------------------------
# Constants (from C++ enums)
# ---------------------------------------------------------------------------

# SIZE_PARAMETERIZATION
SIZE_POSE = 7
SIZE_SPEEDBIAS = 9
SIZE_FEATURE = 1
G = np.array([0.0, 0.0, 9.81007], dtype=np.float64)

# StateOrder
O_P = 0
O_R = 3
O_V = 6
O_BA = 9
O_BG = 12

# NoiseOrder
O_AN = 0
O_GN = 3
O_AW = 6
O_GW = 9


class Parameters(Config):
    """Runtime parameter container for the Python port of VINS‑Fusion."""

    # --------------------------- public API ---------------------------------

    def __init__(self, yaml_file_path: str | os.PathLike[str]):
        super().__init__(yaml_file_path)
        self._read_parameters()

    # -------------------------- private helpers -----------------------------

    def _read_parameters(
        self,
    ) -> None:
        """Parse the main YAML and linked camera calibration YAML."""
        # --- root paths -----------------------------------------------------
        self.config_path = Path(self._yaml_file_path).resolve()
        self.config_dir = self.config_path.parent

        # --------------------------------------------------------------------
        # Dataset (bag / file) paths
        # --------------------------------------------------------------------
        dataset_cfg: Dict = self.get("dataset", {})  # type: ignore[arg-type]
        self.dataset_type: str | None = dataset_cfg.get("type")
        self.input_path: Path | None = (
            Path(dataset_cfg["input_path"]).resolve()
            if "input_path" in dataset_cfg
            else None
        )
        self.output_path: Path | None = (
            Path(dataset_cfg["output_path"]).resolve()
            if "output_path" in dataset_cfg
            else None
        )

        # Convenience default (used by legacy code):
        self.vins_result_path: Path | None = (
            self.output_path / "vio.csv" if self.output_path else None
        )

        # --------------------------------------------------------------------
        # Camera block (image sizing, model, extrinsic, calibration file …)
        # --------------------------------------------------------------------
        camera_cfg: Dict = self.get("camera", {})  # type: ignore[arg-type]

        # Basic geometry / sensor info
        self.camera_comment: str | None = camera_cfg.get("comment")
        self.camera_model: str | None = camera_cfg.get("model")
        self.distortion_model: str | None = camera_cfg.get("distortion_model")
        self.row: int | None = camera_cfg.get("image_height")
        self.col: int | None = camera_cfg.get("image_width")
        self.camera_rate_hz: int | float | None = camera_cfg.get("rate_hz")

        # Calibration behaviour flags
        self.load_previous_calib: bool = bool(
            camera_cfg.get("load_previous_calibration_result", 0)
        )
        # estimate_extrinsic may occasionally be written on a *new line* in the YAML.
        # yaml.safe_load will coerce it into a Python int automatically.
        self.estimate_extrinsic: int = int(camera_cfg.get("estimate_extrinsic", 2))

        # Path to camera calibration YAML
        calib_path_str: str | None = camera_cfg.get("calibration_result_save_path")
        self.calibration_path: Path | None = (
            Path(calib_path_str).resolve() if calib_path_str else None
        )

        # --------------------------------------------------------------------
        # Visual tracking parameters (feature tracker)
        # --------------------------------------------------------------------
        self.max_cnt: int = self.get("max_cnt")
        self.min_dist: float = self.get("min_dist")
        self.freq: int = self.get("freq")
        self.f_threshold: float = self.get("F_threshold")
        self.show_track: bool = bool(self.get("show_track"))
        self.flow_back: bool = bool(self.get("flow_back"))

        # --------------------------------------------------------------------
        # IMU noise parameters
        # --------------------------------------------------------------------
        self.acc_n: float = self.get("acc_n")
        self.gyr_n: float = self.get("gyr_n")
        self.acc_w: float = self.get("acc_w")
        self.gyr_w: float = self.get("gyr_w")
        self.g_norm: float = self.get("g_norm")

        # --------------------------------------------------------------------
        # Solver / estimator parameters
        # --------------------------------------------------------------------
        self.focal_length: float = self.get("focal_length", default=460.0)
        self.window_size: int = self.get("window_size", 10)
        self.solver_time: float = self.get("max_solver_time")
        self.num_iterations: int = self.get("max_num_iterations")
        self.keyframe_parallax: float = self.get("keyframe_parallax")
        self.min_parallax: float = self.keyframe_parallax / self.focal_length

        # Focal length handling: either explicit or derived from camera intrinsics (preferred)

        # --------------------------------------------------------------------
        # Camera calibration specifics (intrinsic & extrinsic)
        # --------------------------------------------------------------------
        (
            self.cam_extrinsic,
            self.cam_intrinsic_matrix,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.distortion_coefficients,
        ) = self._read_camera_config(self.calibration_path)

        # If focal_length unspecified above, derive from calibration
        if self.focal_length is None and self.fx is not None and self.fy is not None:
            self.focal_length = (float(self.fx) + float(self.fy)) / 2.0

        # --------------------------------------------------------------------
        # Extrinsic initialisation for VINS‑Fusion
        #   (body ᵀ camera stored as self.cam_extrinsic above)
        #   The original C++ code stores rotation & translation separately in lists ric[], tic[]
        # --------------------------------------------------------------------
        if self.cam_extrinsic is not None:
            rot = self.cam_extrinsic[:3, :3]
            trans = self.cam_extrinsic[:3, 3]
        else:
            rot = np.eye(3)
            trans = np.zeros(3)

        if self.estimate_extrinsic == 2:
            # No prior knowledge – start from identity but keep loaded calib for save path
            self.ric: List[np.ndarray] = [np.eye(3)]
            self.tic: List[np.ndarray] = [np.zeros(3)]
        else:
            # Use given rotation & translation (and optionally optimise around them)
            self.ric = [rot]
            self.tic = [trans]

        # Where to store the *estimated* extrinsics if calibration is run online
        if self.output_path:
            self.ex_calib_result_path: Path = (
                self.output_path / "extrinsic_parameter.csv"
            )

        # --------------------------------------------------------------------
        # Misc / defaults
        # --------------------------------------------------------------------
        self.gravity_vector = np.array([0.0, 0.0, self.g_norm], dtype=np.float64)
        self.init_depth: float = 5.0
        self.bias_acc_threshold: float = 0.1
        self.bias_gyr_threshold: float = 0.1

    # ................................ reload logic ..........................

    def reload(self) -> None:
        """Thread‑safe re‑parse of both the main config *and* camera YAML."""
        with self._lock:
            self._load()  # Config: re‑read underlying YAML file
            self._read_parameters()

    # ............................... internals ..............................

    def _read_camera_config(self, path: Path | None) -> Tuple[
        np.ndarray | None,
        np.ndarray | None,
        float | None,
        float | None,
        float | None,
        float | None,
        np.ndarray | None,
    ]:
        """Load camera extrinsic & intrinsic parameters from a sensor‑YAML.

        Parameters
        ----------
        path : pathlib.Path | None
            Absolute path to the camera calibration YAML. Can be *None* (e.g. if not provided).

        Returns
        -------
        (extrinsic_4x4, intrinsic_3x3, fx, fy, cx, cy, distortion_coeffs)
        """
        if path is None or not path.exists():
            # Missing file – return None placeholders but keep program running
            return (None, None, None, None, None, None, None)

        cam_dict = _load_yaml(path)

        print(cam_dict)

        # --------------------- extrinsic (T_BS) -----------------------------
        t_bs_cfg = cam_dict.get("T_BS", {})
        extrinsic = _reshape_4x4(t_bs_cfg.get("data", []))

        # --------------------- intrinsics ----------------------------------
        intrinsics_list = cam_dict.get("intrinsics", [])  # [fx, fy, cx, cy]
        if len(intrinsics_list) != 4:
            raise ValueError("Expected 4 numbers in 'intrinsics' list (fx, fy, cx, cy)")
        fx, fy, cx, cy = map(float, intrinsics_list)
        intrinsic_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

        # Distortion coefficients
        dist_coeffs = np.asarray(
            cam_dict.get("distortion_coefficients", []), dtype=np.float64
        )

        return extrinsic, intrinsic_matrix, fx, fy, cx, cy, dist_coeffs
