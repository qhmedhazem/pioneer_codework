import numpy as np
import cv2
from scipy.optimize import leastsq

# Fit scale (a, b)
def fit_scale(dense_depth, sparse_points):
    rel_values = []
    ref_values = []

    for x, y, d in sparse_points:
        rel = dense_depth[int(y), int(x)]
        if rel > 0:
            rel_values.append(rel)
            ref_values.append(d)

    def error_func(params, x, y):
        a, b = params
        return y - (a * np.array(x) + b)

    a_b, _ = leastsq(error_func, [1, 0], args=(rel_values, ref_values))
    return a_b[0], a_b[1]

# Apply metric scale
def apply_scale(dense_depth, a, b):
    return a * dense_depth + b

def warp_depth(depth, K, T):
    """
    depth: (H, W) metric depth map
    K: (3, 3) camera intrinsics
    T: (4, 4) transformation from t-1 to t
    """
    H, W = depth.shape

    y, x = np.indices((H, W))
    z = depth.flatten()
    x = x.flatten()
    y = y.flatten()

    # Backproject to 3D
    pts = np.vstack((x, y, np.ones_like(x)))
    pts = np.linalg.inv(K) @ pts * z

    pts_h = np.vstack((pts, np.ones((1, pts.shape[1]))))
    pts_trans = T @ pts_h
    pts_proj = K @ pts_trans[:3, :]
    pts_proj /= pts_proj[2, :]

    # Reconstruct warped depth map
    x_new = np.round(pts_proj[0, :]).astype(int)
    y_new = np.round(pts_proj[1, :]).astype(int)
    valid = (x_new >= 0) & (x_new < W) & (y_new >= 0) & (y_new < H)

    warped = np.zeros_like(depth)
    warped[y_new[valid], x_new[valid]] = pts_trans[2, valid]
    return warped


def temporal_blending(D_warped, D_current):
    valid = (D_warped > 0) & (D_current > 0)
    if not np.any(valid):
        return D_current

    r = np.mean(np.abs(D_warped[valid] - D_current[valid]) / D_warped[valid])
    alpha = 0.8 if r >= 0.8 else max(0, r)
    return alpha * D_warped + (1 - alpha) * D_current
