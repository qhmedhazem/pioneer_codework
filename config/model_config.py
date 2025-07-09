from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from config.config import Config, _load_yaml, _reshape_4x4


class ModelConfig(Config):
    """Runtime parameter container for the Python port of VINS‑Fusion."""

    def __init__(self, yaml_file_path: str | os.PathLike[str]):
        super().__init__(yaml_file_path)
        self._read_parameters()

    def _read_parameters(
        self,
    ) -> None:
        """Parse the main YAML and linked camera calibration YAML."""
        self.config_path = Path(self._yaml_file_path).resolve()
        self.config_dir = self.config_path.parent

        self.arch = self.get("arch", {})
        self.arch["seed"] = self.arch.get(
            "seed", 42
        )  # Random seed for Pytorch/Numpy initialization
        self.arch["min_epochs"] = self.arch.get(
            "min_epochs", 1
        )  # Minimum number of epochs
        self.arch["max_epochs"] = self.arch.get(
            "max_epochs", 50
        )  # Maximum number of epochs

        self.checkpoint = self.get("checkpoint", {})
        self.checkpoint["filepath"] = self.checkpoint.get("filepath", "")
        self.checkpoint["save_top_k"] = self.checkpoint.get("save_top_k", 5)
        self.checkpoint["monitor"] = self.checkpoint.get("monitor", "loss")
        self.checkpoint["monitor_index"] = self.checkpoint.get("monitor_index", 0)
        self.checkpoint["mode"] = self.checkpoint.get("mode", "auto")
        self.checkpoint["s3_path"] = self.checkpoint.get("s3_path", "")
        self.checkpoint["s3_frequency"] = self.checkpoint.get("s3_frequency", 1)

        self.save = self.get("save", {})
        self.save["folder"] = self.save.get("folder", "")
        self.save["depth"] = self.save.get("depth", {})
        self.save["depth"]["rgb"] = self.save["depth"].get("rgb", True)
        self.save["depth"]["viz"] = self.save["depth"].get("viz", True)
        self.save["depth"]["npz"] = self.save["depth"].get("npz", True)
        self.save["depth"]["png"] = self.save["depth"].get("png", True)

        self.wandb = self.get("wandb", {})
        self.wandb["dry_run"] = self.wandb.get("dry_run", True)
        self.wandb["name"] = self.wandb.get("name", "")
        self.wandb["project"] = self.wandb.get(
            "project", os.environ.get("WANDB_PROJECT", "")
        )
        self.wandb["entity"] = self.wandb.get(
            "entity", os.environ.get("WANDB_ENTITY", "")
        )
        self.wandb["tags"] = self.wandb.get("tags", [])
        self.wandb["dir"] = self.wandb.get("dir", "")

        self.model = self.get("model", {})
        self.model["checkpoint_path"] = self.model.get("checkpoint_path", "")

        self.model["optimizer"] = self.model.get("optimizer", {})
        self.model["optimizer"]["name"] = self.model["optimizer"].get("name", "Adam")
        self.model["optimizer"]["depth"] = self.model["optimizer"].get("depth", {})
        self.model["optimizer"]["depth"]["lr"] = self.model["optimizer"]["depth"].get(
            "lr", 0.0002
        )
        self.model["optimizer"]["depth"]["weight_decay"] = self.model["optimizer"][
            "depth"
        ].get("weight_decay", 0.0)

        self.model["scheduler"] = self.model.get("scheduler", {})
        self.model["scheduler"]["name"] = self.model["scheduler"].get("name", "StepLR")
        self.model["scheduler"]["step_size"] = self.model["scheduler"].get(
            "step_size", 10
        )
        self.model["scheduler"]["gamma"] = self.model["scheduler"].get("gamma", 0.5)
        self.model["scheduler"]["T_max"] = self.model["scheduler"].get("T_max", 20)

        self.model["params"] = self.model.get("params", {})
        self.model["params"]["crop"] = self.model["params"].get("crop", "")
        self.model["params"]["min_depth"] = self.model["params"].get("min_depth", 0.0)
        self.model["params"]["max_depth"] = self.model["params"].get("max_depth", 80.0)
        self.model["params"]["scale_output"] = self.model["params"].get(
            "scale_output", "resize"
        )

        self.model["loss"] = self.model.get("loss", {})
        self.model["loss"]["num_scales"] = self.model["loss"].get("num_scales", 4)
        self.model["loss"]["progressive_scaling"] = self.model["loss"].get(
            "progressive_scaling", 0.0
        )
        self.model["loss"]["flip_lr_prob"] = self.model["loss"].get("flip_lr_prob", 0.5)
        self.model["loss"]["rotation_mode"] = self.model["loss"].get(
            "rotation_mode", "euler"
        )
        self.model["loss"]["upsample_depth_maps"] = self.model["loss"].get(
            "upsample_depth_maps", True
        )
        self.model["loss"]["ssim_loss_weight"] = self.model["loss"].get(
            "ssim_loss_weight", 0.85
        )
        self.model["loss"]["occ_reg_weight"] = self.model["loss"].get(
            "occ_reg_weight", 0.1
        )
        self.model["loss"]["smooth_loss_weight"] = self.model["loss"].get(
            "smooth_loss_weight", 0.001
        )
        self.model["loss"]["C1"] = self.model["loss"].get("C1", 1e-4)
        self.model["loss"]["C2"] = self.model["loss"].get("C2", 9e-4)
        self.model["loss"]["photometric_reduce_op"] = self.model["loss"].get(
            "photometric_reduce_op", "min"
        )
        self.model["loss"]["disp_norm"] = self.model["loss"].get("disp_norm", True)
        self.model["loss"]["clip_loss"] = self.model["loss"].get("clip_loss", 0.0)
        self.model["loss"]["padding_mode"] = self.model["loss"].get(
            "padding_mode", "zeros"
        )
        self.model["loss"]["automask_loss"] = self.model["loss"].get(
            "automask_loss", True
        )

        self.model["depth_net"] = self.model.get("depth_net", {})
        self.model["depth_net"]["checkpoint_path"] = self.model["depth_net"].get(
            "checkpoint_path", ""
        )
        self.model["depth_net"]["version"] = self.model["depth_net"].get("version", "")
        self.model["depth_net"]["dropout"] = self.model["depth_net"].get("dropout", 0.0)

        self.datasets = self.get("datasets", {})
        self.datasets["augmentation"] = self.datasets.get("augmentation", {})
        self.datasets["augmentation"]["image_shape"] = self.datasets[
            "augmentation"
        ].get("image_shape", ())
        self.datasets["augmentation"]["jittering"] = self.datasets["augmentation"].get(
            "jittering", (0.2, 0.2, 0.2, 0.05)
        )
        self.datasets["augmentation"]["crop_train_borders"] = self.datasets[
            "augmentation"
        ].get("crop_train_borders", ())
        self.datasets["augmentation"]["crop_eval_borders"] = self.datasets[
            "augmentation"
        ].get("crop_eval_borders", ())

        self.datasets["train"] = self.datasets.get("train", {})
        self.datasets["train"]["batch_size"] = self.datasets["train"].get(
            "batch_size", 8
        )
        self.datasets["train"]["num_workers"] = self.datasets["train"].get(
            "num_workers", 16
        )
        self.datasets["train"]["back_context"] = self.datasets["train"].get(
            "back_context", 1
        )
        self.datasets["train"]["forward_context"] = self.datasets["train"].get(
            "forward_context", 1
        )
        self.datasets["train"]["dataset"] = self.datasets["train"].get("dataset", [])
        self.datasets["train"]["path"] = self.datasets["train"].get("path", [])
        self.datasets["train"]["split"] = self.datasets["train"].get("split", [])
        self.datasets["train"]["depth_type"] = self.datasets["train"].get(
            "depth_type", [""]
        )
        self.datasets["train"]["input_depth_type"] = self.datasets["train"].get(
            "input_depth_type", [""]
        )
        self.datasets["train"]["cameras"] = self.datasets["train"].get("cameras", [[]])
        self.datasets["train"]["repeat"] = self.datasets["train"].get("repeat", [1])
        self.datasets["train"]["num_logs"] = self.datasets["train"].get("num_logs", 5)

        self.datasets["validation"] = self.datasets.get("validation", {})
        self.datasets["validation"]["batch_size"] = self.datasets["validation"].get(
            "batch_size", 1
        )
        self.datasets["validation"]["num_workers"] = self.datasets["validation"].get(
            "num_workers", 8
        )
        self.datasets["validation"]["back_context"] = self.datasets["validation"].get(
            "back_context", 0
        )
        self.datasets["validation"]["forward_context"] = self.datasets[
            "validation"
        ].get("forward_context", 0)
        self.datasets["validation"]["dataset"] = self.datasets["validation"].get(
            "dataset", []
        )
        self.datasets["validation"]["path"] = self.datasets["validation"].get(
            "path", []
        )
        self.datasets["validation"]["split"] = self.datasets["validation"].get(
            "split", []
        )
        self.datasets["validation"]["depth_type"] = self.datasets["validation"].get(
            "depth_type", [""]
        )
        self.datasets["validation"]["input_depth_type"] = self.datasets[
            "validation"
        ].get("input_depth_type", [""])
        self.datasets["validation"]["cameras"] = self.datasets["validation"].get(
            "cameras", [[]]
        )
        self.datasets["validation"]["num_logs"] = self.datasets["validation"].get(
            "num_logs", 5
        )

        self.datasets["test"] = self.datasets.get("test", {})
        self.datasets["test"]["batch_size"] = self.datasets["test"].get("batch_size", 1)
        self.datasets["test"]["num_workers"] = self.datasets["test"].get(
            "num_workers", 8
        )
        self.datasets["test"]["back_context"] = self.datasets["test"].get(
            "back_context", 0
        )
        self.datasets["test"]["forward_context"] = self.datasets["test"].get(
            "forward_context", 0
        )
        self.datasets["test"]["dataset"] = self.datasets["test"].get("dataset", [])
        self.datasets["test"]["path"] = self.datasets["test"].get("path", [])
        self.datasets["test"]["split"] = self.datasets["test"].get("split", [])
        self.datasets["test"]["depth_type"] = self.datasets["test"].get(
            "depth_type", [""]
        )
        self.datasets["test"]["input_depth_type"] = self.datasets["test"].get(
            "input_depth_type", [""]
        )
        self.datasets["test"]["cameras"] = self.datasets["test"].get("cameras", [[]])
        self.datasets["test"]["num_logs"] = self.datasets["test"].get("num_logs", 5)

        self.config = self.get("config", "")
        self.default = self.get("default", "")
        self.wandb["url"] = self.wandb.get("url", "")
        self.checkpoint["s3_url"] = self.checkpoint.get("s3_url", "")
        self.save["pretrained"] = self.save.get("pretrained", "")
        self.prepared = self.get("prepared", False)

    def reload(self) -> None:
        """Thread‑safe re‑parse of both the main config *and* camera YAML."""
        with self._lock:
            self._load()
            self._read_parameters()

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

        t_bs_cfg = cam_dict.get("T_BS", {})
        extrinsic = _reshape_4x4(t_bs_cfg.get("data", []))

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
