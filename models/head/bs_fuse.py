# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, NonLocal2d

from .base_decoder import resize
from .base_decoder import BaseDecodeHead
from einops import rearrange

up_kwargs = {'mode': 'bilinear', 'align_corners': False}
norm_cfg = dict(type='BN', requires_grad=True)


class BSFuseHead(nn.Module):
    def __init__(self, in_channels=[128, 512], in_index=[0, 1], num_classes=2, channels=128):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.nclass = num_classes
        self.low_level_fuse = nn.Conv2d(self.in_channels[1], self.channels, 1)
        self.pre = nn.Conv2d(self.channels, self.nclass, 1)


    def forward(self, x, edge):
        out = edge + self.low_level_fuse(x)
        # ========= output ==============
        out = self.pre(out)
        return out
