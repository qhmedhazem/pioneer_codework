import torch
import torch.nn as nn
import random

from networks.depth_network import DepthNet
from geometry.pose import Pose
from losses.photometric_loss import PhotometricLoss
from lib.model_helpers import (
    flip_batch_input,
    flip_output,
    upsample_output,
    merge_outputs,
)
from packnet_sfm.utils.misc import filter_dict


class SemiSupModel(nn.Module):
    """
    Self-supervised depth estimation model using externally provided poses (from VIO).

    Parameters
    ----------
    depth_net : nn.Module
        Depth network to be used
    flip_lr_prob : float
        Probability of flipping when using the depth network
    upsample_depth_maps : bool
        True if depth map scales are upsampled to highest resolution
    kwargs : dict
        Extra parameters (e.g. photometric loss config)
    """

    def __init__(self, flip_lr_prob=0.0, upsample_depth_maps=False, **kwargs):
        super().__init__()
        self.depth_net = DepthNet(**kwargs)
        self.flip_lr_prob = flip_lr_prob
        self.upsample_depth_maps = upsample_depth_maps

        self._logs = {}
        self._losses = {}
        self._photometric_loss = PhotometricLoss(**kwargs)

    @property
    def logs(self):
        return {**self._logs, **self._photometric_loss.logs}

    @property
    def losses(self):
        return self._losses

    def add_loss(self, key, val):
        self._losses[key] = val.detach()

    def depth_net_flipping(self, batch, flip):
        batch_input = {key: batch[key] for key in filter_dict(batch, ["rgb"])}
        if flip:
            output = self.depth_net(**flip_batch_input(batch_input))
            output = flip_output(output)
        else:
            output = self.depth_net(**batch_input)
        return output

    def compute_depth_net(self, batch, force_flip=False):
        flag_flip_lr = (
            random.random() < self.flip_lr_prob if self.training else force_flip
        )
        output = self.depth_net_flipping(batch, flag_flip_lr)
        if self.training and self.upsample_depth_maps:
            output = upsample_output(output, mode="nearest", align_corners=None)
        return output

    def forward(self, batch, return_logs=False, progress=0.0):
        output = self.compute_depth_net(batch, force_flip=False)

        if not self.training:
            return output

        self_sup_output = self._photometric_loss(
            batch["rgb_original"],
            batch["rgb_context_original"],
            output["inv_depths"],
            batch["intrinsics"],
            batch["intrinsics"],
            [Pose.from_numpy(p) for p in batch["poses"]],
            return_logs=return_logs,
            progress=progress,
        )

        return {
            "loss": self_sup_output["loss"],
            **merge_outputs(output, self_sup_output),
        }
