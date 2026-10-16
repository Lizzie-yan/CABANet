# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, NonLocal2d

from .base_decoder import resize
from .base_decoder import BaseDecodeHead

up_kwargs = {'mode': 'bilinear', 'align_corners': False}
norm_cfg = dict(type='BN', requires_grad=True)

# class FuseHead(BaseDecodeHead):
#     def __init__(self, in_channels=[256, 512, 1024, 2048], in_index=[0, 1, 2, 3], num_classes=2, channels=512):
#         super().__init__(input_transform='multiple_select',
#                          in_channels=in_channels, in_index=[0, 1, 2, 3], num_classes=num_classes,
#                          channels=512, dropout_ratio=0.1, norm_cfg=norm_cfg, align_corners=False)
#         num_inputs = len(self.in_channels)
#         assert num_inputs == len(self.in_index)