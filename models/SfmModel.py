# Copyright 2020 Toyota Research Institute.  All rights reserved.

import random

from geometry.pose import Pose
from models.base_model import BaseModel
from models.model_utils import flip_batch_input, flip_output, upsample_output
from utils.misc import filter_dict


class SfmModel(BaseModel):
    """
    Model class encapsulating a pose and depth networks.

    Parameters
    ----------
    depth_net : nn.Module
        Depth network to be used
    pose_net : nn.Module
        Pose network to be used
    rotation_mode : str
        Rotation mode for the pose network
    flip_lr_prob : float
        Probability of flipping when using the depth network
    upsample_depth_maps : bool
        True if depth map scales are upsampled to highest resolution
    kwargs : dict
        Extra parameters
    """
    def __init__(self, depth_net=None, pose_net=None,
                 rotation_mode='euler', flip_lr_prob=0.0,
                 upsample_depth_maps=False, **kwargs):
        super().__init__()
        self.depth_net = depth_net
        self.rotation_mode = rotation_mode
        self.flip_lr_prob = flip_lr_prob
        self.upsample_depth_maps = upsample_depth_maps

        self._network_requirements = [
            'depth_net',
        ]

    def add_depth_net(self, depth_net):
        """Add a depth network to the model"""
        self.depth_net = depth_net


    def depth_net_flipping(self, batch, flip):
        """
        Runs depth net with the option of flipping

        Parameters
        ----------
        batch : dict
            Input batch
        flip : bool
            True if the flip is happening

        Returns
        -------
        output : dict
            Dictionary with depth network output (e.g. 'inv_depths' and 'uncertainty')
        """
        # Which keys are being passed to the depth network
        batch_input = {key: batch[key] for key in filter_dict(batch, self._input_keys)}
        if flip:
            # Run depth network with flipped inputs
            output = self.depth_net(**flip_batch_input(batch_input))
            # Flip output back if training
            output = flip_output(output)
        else:
            # Run depth network
            output = self.depth_net(**batch_input)
        return output

    def compute_depth_net(self, batch, force_flip=False):
        """Computes inverse depth maps from single images"""
        # Randomly flip and estimate inverse depth maps
        flag_flip_lr = random.random() < self.flip_lr_prob if self.training else force_flip
        output = self.depth_net_flipping(batch, flag_flip_lr)
        # If upsampling depth maps at training time
        if self.training and self.upsample_depth_maps:
            output = upsample_output(output, mode='nearest', align_corners=None)
        # Return inverse depth maps
        return output

    def get_poses(self, context):
        """Compute poses from image and a sequence of context images"""
        return [Pose.from_vec(context[i], None)
                for i in range(len(context))]

    def forward(self, batch, return_logs=False, force_flip=False):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored
        force_flip : bool
            If true, force batch flipping for inverse depth calculation

        Returns
        -------
        output : dict
            Dictionary containing the output of depth and pose networks
        """
        # Generate inverse depth predictions
        depth_output = self.compute_depth_net(batch, force_flip=force_flip)
        # Generate pose predictions if available
        pose_output = None
        if 'pose_context' in batch:
            pose_output = self.get_poses(batch['pose_context'])
        # Return output dictionary
        return {
            **depth_output,
            'poses': pose_output,
        }
