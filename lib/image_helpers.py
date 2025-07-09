import cv2
import torch
import torch.nn.functional as funct
from functools import lru_cache
from PIL import Image


def same_shape(shape1, shape2):
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if shape1[i] != shape2[i]:
            return False
    return True


def load_image(path):
    return Image.open(path)


def crop_to_aspect_ratio(image, target_shape):
    orig_width, orig_height = image.size
    orig_ratio = orig_width / orig_height
    target_height, target_width = target_shape
    target_ratio = target_width / target_height
    if orig_ratio > target_ratio:
        new_width = int(orig_height * target_ratio)
        new_height = orig_height
        left = (orig_width - new_width) // 2
        top = 0
    else:
        new_width = orig_width
        new_height = int(orig_width / target_ratio)
        left = 0
        top = (orig_height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    return image.crop((left, top, right, bottom))


def write_image(filename, image):
    cv2.imwrite(filename, image[:, :, ::-1])


def flip_lr(image):
    assert image.dim() == 4, "You need to provide a [B,C,H,W] image to flip"
    return torch.flip(image, [3])


def flip_model(model, image, flip):
    if flip:
        return [flip_lr(inv_depth) for inv_depth in model(flip_lr(image))]
    else:
        return model(image)


def gradient_x(image):
    return image[:, :, :, :-1] - image[:, :, :, 1:]


def gradient_y(image):
    return image[:, :, :-1, :] - image[:, :, 1:, :]


def interpolate_image(image, shape, mode="bilinear", align_corners=True):
    if len(shape) > 2:
        shape = shape[-2:]
    if same_shape(image.shape[-2:], shape):
        return image
    else:
        return funct.interpolate(
            image, size=shape, mode=mode, align_corners=align_corners
        )


def interpolate_scales(images, shape=None, mode="bilinear", align_corners=False):
    if shape is None:
        shape = images[0].shape
    if len(shape) > 2:
        shape = shape[-2:]
    return [
        funct.interpolate(image, shape, mode=mode, align_corners=align_corners)
        for image in images
    ]


def match_scales(image, targets, num_scales, mode="bilinear", align_corners=True):
    images = []
    image_shape = image.shape[-2:]
    for i in range(num_scales):
        target_shape = targets[i].shape
        if same_shape(image_shape, target_shape):
            images.append(image)
        else:
            images.append(
                interpolate_image(
                    image, target_shape, mode=mode, align_corners=align_corners
                )
            )
    return images


@lru_cache(maxsize=None)
def meshgrid(B, H, W, dtype, device, normalized=False):
    if normalized:
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    else:
        xs = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
        ys = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
    ys, xs = torch.meshgrid([ys, xs])
    return xs.repeat([B, 1, 1]), ys.repeat([B, 1, 1])


@lru_cache(maxsize=None)
def image_grid(B, H, W, dtype, device, normalized=False):
    xs, ys = meshgrid(B, H, W, dtype, device, normalized=normalized)
    ones = torch.ones_like(xs)
    grid = torch.stack([xs, ys, ones], dim=1)
    return grid
