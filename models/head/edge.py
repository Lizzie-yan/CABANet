import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule, NonLocal2d
from .base_decoder import resize
from tools.heatmap_fun import draw_features

up_kwargs = {'mode': 'bilinear', 'align_corners': False}

def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)
    
    # 1. 创建卷积层
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)

    # 2. 使用 with torch.no_grad() 来修改参数数据，这更安全
    with torch.no_grad():
        # 直接修改 .data，这是最稳妥的方式
        conv_x.weight.copy_(torch.from_numpy(filter_x))
        conv_y.weight.copy_(torch.from_numpy(filter_y))
    
    # 3. 如果想让这个参数不可训练，可以设置它的 requires_grad 属性
    conv_x.weight.requires_grad = False
    conv_y.weight.requires_grad = False

    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y

def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input


class SCSEAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.fusion_conv = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1) #增加一个1*1的卷积
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        x_cSE = x * self.cSE(x)
        x_sSE = x * self.sSE(x)
        concatenated_features = torch.cat([x_cSE, x_sSE], dim=1)
        fused_features = self.fusion_conv(concatenated_features)
        return fused_features     #特征融合
        #return x * self.cSE(x)   + x * self.sSE(x)

class ReverseEdge(nn.Module):
    def __init__(self, channel):
        super().__init__()

        # self.edge_pred = nn.Conv2d(
        #     channel, channel,
        #     kernel_size=3, stride=1,
        #     padding=1, bias=False)
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, c_fuse):
        avg_high = torch.mean(c_fuse, dim=1, keepdim=True)
        max_high, _ = torch.max(c_fuse, dim=1, keepdim=True)
        x = torch.cat([avg_high, max_high], dim=1)
        x = self.conv1(x).sigmoid()

        return c_fuse * x

class EdgeHead(nn.Module):
    """Edge awareness module = Boundary Awareness Enhancement Module.

    三个单元可通过开关逐步启用，用于单元级消融：
      use_bau  : Boundary Awareness Unit       —— Sobel + 两个 3x3 卷积 + 残差
      use_ceu  : Contextual Enhancement Unit   —— 深层语义特征引导增强
      use_bffu : Boundary Feature Fusion Unit  —— low_fuse + SCSE 注意力融合
    三个都开启时与原实现完全等价（模块名/参数形状不变）。
    """

    def __init__(self, in_channels=[96, 192], channels=96, out_fea=2, in_index=[0, 1],
                 use_bau=True, use_ceu=True, use_bffu=True):
        super(EdgeHead, self).__init__()
        self.in_index = in_index
        self.in_channels = in_channels
        self.channels = channels
        self.use_bau = use_bau
        self.use_ceu = use_ceu
        self.use_bffu = use_bffu
        self.norm_cfg = dict(type='BN', requires_grad=True)
        self.act_cfg = dict(type='ReLU')

        # --- 基础层：三种配置都用 ---
        self.conv_base1 = ConvModule(
            in_channels=self.in_channels[0], out_channels=self.channels,
            kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv_base2 = ConvModule(
            in_channels=self.in_channels[1], out_channels=self.channels,
            kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv_edge = nn.Conv2d(self.channels, out_fea, 1, 1, 0)

        # --- 单元 1：Boundary Awareness Unit ---
        if self.use_bau:
            self.conv_base01 = ConvModule(
                in_channels=self.in_channels[0], out_channels=self.channels,
                kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            self.conv_fuse1 = nn.Sequential(
                ConvModule(in_channels=self.channels, out_channels=self.channels,
                           kernel_size=3, padding=1, norm_cfg=self.norm_cfg,
                           act_cfg=self.act_cfg),
                ConvModule(in_channels=self.channels, out_channels=self.channels,
                           kernel_size=3, padding=1, norm_cfg=self.norm_cfg,
                           act_cfg=self.act_cfg))
            self.conv_base02 = ConvModule(
                in_channels=self.in_channels[1], out_channels=self.channels,
                kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            self.conv_fuse2 = nn.Sequential(
                ConvModule(in_channels=self.channels, out_channels=self.channels,
                           kernel_size=3, padding=1, norm_cfg=self.norm_cfg,
                           act_cfg=self.act_cfg),
                ConvModule(in_channels=self.channels, out_channels=self.channels,
                           kernel_size=3, padding=1, norm_cfg=self.norm_cfg,
                           act_cfg=self.act_cfg))
            self.sobel_x1, self.sobel_y1 = get_sobel(256, 1)
            self.sobel_x2, self.sobel_y2 = get_sobel(512, 1)

        # --- 单元 2：Contextual Enhancement Unit ---
        if self.use_ceu:
            self.global_embedding11 = ConvModule(
                in_channels=512, out_channels=self.channels, kernel_size=1,
                norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            self.global_embedding12 = ConvModule(
                in_channels=512, out_channels=self.channels, kernel_size=1,
                norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            self.global_embedding21 = ConvModule(
                in_channels=512, out_channels=self.channels, kernel_size=1,
                norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            self.global_embedding22 = ConvModule(
                in_channels=512, out_channels=self.channels, kernel_size=1,
                norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        # --- 单元 3：Boundary Feature Fusion Unit ---
        if self.use_bffu:
            self.low_fuse = ConvModule(
                in_channels=self.channels * 2, out_channels=self.channels,
                kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            self.scse = SCSEAttention(in_channels=self.channels)
        else:
            self.concat_fuse = ConvModule(
                in_channels=self.channels * 2, out_channels=self.channels,
                kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        self.act = nn.Sigmoid()

    def _transform_inputs(self, inputs):
        if isinstance(self.in_index, (list, tuple)):
            inputs = [inputs[i] for i in self.in_index]
        elif isinstance(self.in_index, int):
            inputs = inputs[self.in_index]
        return inputs

    def _branch(self, c, conv_base, sobel_x=None, sobel_y=None, conv_fuse=None):
        """BAU 开关作用在单条支路上。"""
        if self.use_bau:
            c_b = run_sobel(sobel_x, sobel_y, c)
            c_t = conv_base(c_b)
            c_b = conv_fuse(c_t)
            return c_t + c_b
        return conv_base(c)

    def forward(self, inputs, x_semantic):
        inputs = self._transform_inputs(inputs)
        c1, c2 = inputs  # 1/4 与 1/8

        # ---- 单元1 BAU ----
        if self.use_bau:
            c1_b = self._branch(c1, self.conv_base1, self.sobel_x1, self.sobel_y1,
                                self.conv_fuse1)
        else:
            c1_b = self._branch(c1, self.conv_base1)

        # ---- 单元2 CEU ----
        if self.use_ceu:
            s1 = self.global_embedding12(x_semantic)
            s1 = resize(s1, size=c1.size()[2:], mode='bilinear', align_corners=False)
            c1_b = c1_b * s1 + s1

        if self.use_bau:
            c2_b = self._branch(c2, self.conv_base2, self.sobel_x2, self.sobel_y2,
                                self.conv_fuse2)
        else:
            c2_b = self._branch(c2, self.conv_base2)

        if self.use_ceu:
            s2 = self.global_embedding22(x_semantic)
            s2 = F.interpolate(s2, size=c2_b.shape[2:], mode='bilinear',
                               align_corners=False)
            c2_b = c2_b * s2 + s2

        c2_b = resize(c2_b, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # ---- 单元3 BFFU ----
        cat = torch.cat([c1_b, c2_b], dim=1)
        if self.use_bffu:
            c_fuse = self.scse(self.low_fuse(cat))
        else:
            c_fuse = self.concat_fuse(cat)

        edge = self.conv_edge(c_fuse)
        return edge, c_fuse
