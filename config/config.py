import yaml
import numpy as np
from pathlib import Path
import threading
from typing import Any, Optional, Dict, List


def _load_yaml(path: Path) -> Dict:
    """Utility: safe‑load a YAML file and return a dict."""
    with open(path, "r", encoding="utf‑8") as f:
        return yaml.safe_load(f)


def _reshape_4x4(data: List[float]) -> np.ndarray:
    """Cast a length‑16 flat list to a 4×4 float64 NumPy matrix."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.size != 16:
        raise ValueError("T_BS data does not contain 16 numbers (4×4 matrix)")
    return arr.reshape(4, 4)


class Config:
    """
    Thread-safe YAML configuration handler for mono-camera + IMU applications.

    Provides methods to retrieve and update configuration values in a YAML file.
    """

    def __init__(self, yaml_file_path: str) -> None:
        """
        Initialize the Config by loading data from the given YAML file path.

        :param yaml_file_path: Path to the YAML configuration file.
        """
        self._yaml_file_path = yaml_file_path
        self._lock = threading.RLock()
        self._data: dict = {}
        self._load()

    def _load(self) -> None:
        """
        Load the entire YAML file into an internal dictionary.
        """
        with open(self._yaml_file_path, "r") as file_handle:
            self._data = yaml.safe_load(file_handle) or {}

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieve a configuration value by key.

        :param key: The configuration key to look up.
        :param default: The default value to return if key is not present.
        :return: The configuration value, or default if not found.
        """
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Update a configuration value and immediately save to the YAML file.

        :param key: The configuration key to update.
        :param value: The new value to assign.
        """
        with self._lock:
            self._data[key] = value
            self._save()

    def _save(self) -> None:
        """
        Write the current internal dictionary back to the YAML file.
        """
        with open(self._yaml_file_path, "w") as file_handle:
            yaml.safe_dump(self._data, file_handle)
