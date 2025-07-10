import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from mpl_toolkits.mplot3d import Axes3D  # ← must include this


class TrajectoryViewer:
    """
    Live 3-D trajectory plotter (Matplotlib, no ROS/Open3D needed).
    Call add_pose(T_w_c) with a 4×4 homogeneous camera pose in world frame.
    """
    def __init__(self, buffer_len: int = 1000) -> None:
        self.poses = deque(maxlen=buffer_len)          # type: deque[np.ndarray]
        plt.ion()
        self.fig = plt.figure(figsize=(6, 6))
        self.ax  = self.fig.add_subplot(projection="3d")
        self._set_labels()

    # ------------------------------------------------------------------
    def add_pose(self, T_w_c: np.ndarray) -> None:
        """Append a new pose and refresh the plot."""
        T_w_c = np.asarray(T_w_c)
        if T_w_c.shape != (4, 4):
            raise ValueError("T_w_c must be a 4×4 homogeneous matrix")
        self.poses.append(T_w_c[:3, 3].copy())         # store (x,y,z)
        self._redraw()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _redraw(self) -> None:
        if not self.poses:                             # safety (first frame)
            return
        xs, ys, zs = zip(*self.poses)

        self.ax.clear()
        self.ax.plot(xs, ys, zs, lw=2)
        self.ax.scatter(xs[-1], ys[-1], zs[-1],
                        s=40, marker="o", label="Current")
        self._set_labels()
        self.ax.legend(loc="upper left", frameon=False)
        self.ax.set_xlim(min(xs), max(xs))
        self.ax.set_ylim(min(ys), max(ys))
        self.ax.set_zlim(min(zs), max(zs))

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def _set_labels(self) -> None:
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")
