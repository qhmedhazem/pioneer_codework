import torch
import torch.nn.functional as funct
import numpy as np


def construct_K(fx=0, fy=0, cx=0, cy=0, dtype=torch.float, device=None):
    return torch.tensor(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=dtype, device=device
    )


def construct_from_array(arr, dtype=torch.float, device=None):
    return torch.tensor(arr, dtype=dtype, device=device)


def load_intrinsics(npz_path):
    data = np.load(npz_path)
    K = data.get("K")
    D = data.get("D")
    res = data.get("res")
    return K, D, res


def scale_intrinsics(K, x_scale, y_scale):
    K[..., 0, 0] *= x_scale
    K[..., 1, 1] *= y_scale
    K[..., 0, 2] = (K[..., 0, 2] + 0.5) * x_scale - 0.5
    K[..., 1, 2] = (K[..., 1, 2] + 0.5) * y_scale - 0.5
    return K


def view_synthesis(
    ref_image, depth, ref_cam, cam, mode="bilinear", padding_mode="zeros"
):
    assert depth.size(1) == 1
    world_points = cam.reconstruct(depth, frame="w")
    ref_coords = ref_cam.project(world_points, frame="w")

    ref_coords[..., 0] = (
        ref_coords[..., 0] / ref_coords[..., 2] * ref_image.shape[3] - 0.5
    )  # scale x-coordinates to image space

    return funct.grid_sample(
        ref_image, ref_coords, mode=mode, padding_mode=padding_mode, align_corners=True
    )


def view_synthesis_generic(
    ref_image, depth, ref_cam, cam, mode="bilinear", padding_mode="zeros", progress=0.0
):
    assert depth.size(1) == 1
    world_points = cam.reconstruct(depth, frame="w")
    ref_coords = ref_cam.project(world_points, progress=progress, frame="w")
    return funct.grid_sample(
        ref_image, ref_coords, mode=mode, padding_mode=padding_mode, align_corners=True
    )
