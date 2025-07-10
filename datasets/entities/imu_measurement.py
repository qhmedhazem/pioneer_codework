import numpy as np
from dataclasses import dataclass


@dataclass
class ImuMeasurement:
    """A single IMU sample (gyroscope + accelerometer)."""

    timestamp: float  # seconds
    angular_velocity: np.ndarray  # (3,) rad s‑1
    linear_acceleration: np.ndarray  # (3,) m s‑2
