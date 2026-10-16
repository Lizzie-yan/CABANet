# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, NonLocal2d

from .base_decoder import resize
from .base_decoder import BaseDecodeHead
from einops import rearrange
from tools.heatmap_fun import draw_features

up_kwargs = {'mode': 'bilinear', 'align_corners': False}
norm_cfg = dict(type='BN', requires_grad=True)

import torch
import torch.nn as nn

# class Decomposed3DAttention(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.channels = channels
        
#         # 每个平面的注意力模块
#         self.dh_attention = PlaneAttention(channels)  # D-H平面
#         self.dw_attention = PlaneAttention(channels)  # D-W平面  
#         self.hw_attention = PlaneAttention(channels)  # H-W平面
        
#         # 融合权重
#         self.fusion_conv = nn.Conv3d(channels * 3, channels, kernel_size=1)
#         self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        
#     def forward(self, x):
#         b, c, d, h, w = x.shape
        
#         # --- D-H 平面注意力 ---
#         # 将W维度作为batch维度，处理D-H平面
#         x_dh = x.permute(0, 4, 1, 2, 3).reshape(b * w, c, d, h)  # (B*W, C, D, H)
#         attn_dh = self.dh_attention(x_dh)  # (B*W, C, D, H)
#         attn_dh = attn_dh.reshape(b, w, c, d, h).permute(0, 2, 3, 4, 1)  # (B, C, D, H, W)
        
#         # --- D-W 平面注意力 ---
#         # 将H维度作为batch维度，处理D-W平面
#         x_dw = x.permute(0, 3, 1, 2, 4).reshape(b * h, c, d, w)  # (B*H, C, D, W)
#         attn_dw = self.dw_attention(x_dw)  # (B*H, C, D, W)
#         attn_dw = attn_dw.reshape(b, h, c, d, w).permute(0, 2, 3, 1, 4)  # (B, C, D, H, W)
        
#         # --- H-W 平面注意力 ---
#         # 将D维度作为batch维度，处理H-W平面
#         x_hw = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)  # (B*D, C, H, W)
#         attn_hw = self.hw_attention(x_hw)  # (B*D, C, H, W)
#         attn_hw = attn_hw.reshape(b, d, c, h, w).permute(0, 2, 1, 3, 4)  # (B, C, D, H, W)
        
#         # 融合三个平面的注意力结果
#         combined = torch.cat([attn_dh, attn_dw, attn_hw], dim=1)  # (B, 3*C, D, H, W)
#         out = self.fusion_conv(combined)  # (B, C, D, H, W)
#         out = self.norm(out)
        
#         # 残差连接
#         return out + x


class PlaneAttention(nn.Module):
    """单个平面的注意力模块"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 查询、键、值的投影
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        
        # 输出投影
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # 生成查询、键、值
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # (B, HW, C//8)
        key = self.key_conv(x).view(batch_size, -1, height * width)  # (B, C//8, HW)
        value = self.value_conv(x).view(batch_size, -1, height * width)  # (B, C, HW)
        
        # 计算注意力权重
        attention = torch.bmm(query, key)  # (B, HW, HW)
        attention = F.softmax(attention, dim=-1)
        
        # 应用注意力
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, HW)
        out = out.view(batch_size, channels, height, width)
        
        # 输出投影和残差连接
        out = self.out_conv(out)
        out = self.gamma * out + x
        
        return out

class Decomposed3DAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 用于三个平面的1x1卷积和归一化
        self.conv_dh = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_wd = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_hw = nn.Conv2d(channels, channels, kernel_size=1)
        self.gn = nn.GroupNorm(channels, channels)
        
    def forward(self, x):
        b, c, d, h, w = x.shape
        
        # --- D-H 平面注意力 (复杂度 O(D²H)) ---
        x_dh = x.permute(0, 1, 3, 2, 4).reshape(b * w, c, h, d)  # (B*W, C, H, D)
        attn_dh = torch.sigmoid(self.conv_dh(x_dh.mean(dim=3, keepdim=True)))  # (B*W, C, H, 1)
        x_dh = x_dh * attn_dh  # 广播乘法
        x_dh = x_dh.reshape(b, w, c, h, d).permute(0, 2, 4, 3, 1)  # (B, C, D, H, W)
        
        # --- W-D 平面注意力 (复杂度 O(W²D)) ---
        x_wd = x.permute(0, 1, 2, 4, 3).reshape(b * h, c, d, w)  # (B*H, C, D, W)
        attn_wd = torch.sigmoid(self.conv_wd(x_wd.mean(dim=2, keepdim=True)))  # (B*H, C, 1, W)
        x_wd = x_wd * attn_wd
        x_wd = x_wd.reshape(b, h, c, d, w).permute(0, 2, 3, 1, 4)  # (B, C, D, H, W)
        
        # --- H-W 平面注意力 (复杂度 O(H²W)) ---
        x_hw = x.reshape(b * d, c, h, w)  # (B*D, C, H, W)
        attn_hw = torch.sigmoid(self.conv_hw(x_hw.mean(dim=2, keepdim=True)))  # (B*D, C, 1, W)
        x_hw = x_hw * attn_hw
        x_hw = x_hw.reshape(b, d, c, h, w).permute(0, 2, 1, 3, 4)  # (B, C, D, H, W)
        
        # 融合三个平面的注意力结果
        out = self.gn(x_dh * x_wd * x_hw)  # 逐元素乘法融合
        return out

# 简化版本
# class Decomposed3DAttention(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.channels = channels
        
#         # 每个平面的简单注意力
#         self.dh_conv = nn.Sequential(
#             nn.Conv2d(channels, channels // 4, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // 4, channels, kernel_size=1),
#             nn.Sigmoid()
#         )
        
#         self.dw_conv = nn.Sequential(
#             nn.Conv2d(channels, channels // 4, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // 4, channels, kernel_size=1),
#             nn.Sigmoid()
#         )
        
#         self.hw_conv = nn.Sequential(
#             nn.Conv2d(channels, channels // 4, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // 4, channels, kernel_size=1),
#             nn.Sigmoid()
#         )
        
#         self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
#         self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        
#     def forward(self, x):
#         b, c, d, h, w = x.shape
        
#         # --- D-H 平面注意力 ---
#         x_dh = x.permute(0, 4, 1, 2, 3).reshape(b * w, c, d, h)  # (B*W, C, D, H)
#         attn_dh = self.dh_conv(x_dh)  # (B*W, C, D, H)
#         x_dh_out = x_dh * attn_dh
#         x_dh_out = x_dh_out.reshape(b, w, c, d, h).permute(0, 2, 3, 4, 1)  # (B, C, D, H, W)
        
#         # --- D-W 平面注意力 ---
#         x_dw = x.permute(0, 3, 1, 2, 4).reshape(b * h, c, d, w)  # (B*H, C, D, W)
#         attn_dw = self.dw_conv(x_dw)  # (B*H, C, D, W)
#         x_dw_out = x_dw * attn_dw
#         x_dw_out = x_dw_out.reshape(b, h, c, d, w).permute(0, 2, 3, 1, 4)  # (B, C, D, H, W)
        
#         # --- H-W 平面注意力 ---
#         x_hw = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)  # (B*D, C, H, W)
#         attn_hw = self.hw_conv(x_hw)  # (B*D, C, H, W)
#         x_hw_out = x_hw * attn_hw
#         x_hw_out = x_hw_out.reshape(b, d, c, h, w).permute(0, 2, 1, 3, 4)  # (B, C, D, H, W)
        
#         # 加权融合
#         weights = F.softmax(self.fusion_weights, dim=0)
#         out = weights[0] * x_dh_out + weights[1] * x_dw_out + weights[2] * x_hw_out
#         out = self.norm(out)
        
#         return out + x  # 残差连接
    
# class Decomposed3DAttention(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.channels = channels
        
#         # 用于三个平面的1x1卷积和归一化
#         self.conv_dh = nn.Conv2d(channels, channels, kernel_size=1)
#         self.conv_wd = nn.Conv2d(channels, channels, kernel_size=1)
#         self.conv_hw = nn.Conv2d(channels, channels, kernel_size=1)
#         self.gn = nn.GroupNorm(channels, channels)
        
#     def forward(self, x):
#         b, c, d, h, w = x.shape
        
#         # --- H-W 平面注意力 (复杂度 O(H²W)) ---
#         x_dh = x.permute(0, 1, 3, 2, 4).reshape(b*w , c, h, d) 
#         attn_dh = torch.sigmoid(self.conv_dh(x_dh.mean(dim=3, keepdim=True)))  # (B*W, C, H, 1)
#         x_dh = x_dh * attn_dh  # 广播乘法
#         x_dh = x_dh.reshape(b, w, c, h, d).permute(0, 2, 4, 3, 1)  # (B, C, D, H, W)
        
#         # --- D-H 平面注意力 (复杂度 O(D²H)) ---
#         x_wd = x.permute(0, 1, 2, 4, 3).reshape(b*h, c, d, w)  
#         attn_wd = torch.sigmoid(self.conv_wd(x_wd.mean(dim=3, keepdim=True)))  # (B*H, C, d, 1)
#         x_wd = x_wd * attn_wd
#         x_wd = x_wd.reshape(b, h, c, d, w).permute(0, 2, 3, 1, 4)  # (B, C, D, H, W)
        
#         # --- W-D 平面注意力 (复杂度 O(W²D)) ---
#         x_hw = x.permute(0, 1, 4, 3, 2).reshape(b*d, c, w, h) 
#         attn_hw = torch.sigmoid(self.conv_hw(x_hw.mean(dim=3, keepdim=True)))  # (B*D, C, W, 1)
#         x_hw = x_hw * attn_hw
#         x_hw = x_hw.reshape(b, d, c, w, h).permute(0, 2, 1, 4, 3)  # (B, C, D, H, W)
        
#         # 融合三个平面的注意力结果
#         out = self.gn(x_dh )#* x_wd * x_hw)  # 逐元素乘法融合
#         return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ContAttn(NonLocal2d):
    def __init__(self, *arg, head=1,
                 patch_size=None, **kwargs):
        super().__init__(*arg, **kwargs)
        self.head = head
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(self.in_channels)
        
        # 1. 先创建一个空的 ParameterDict
        self.position_mixing = nn.ParameterDict()
        
        # 2. 使用方括号[]逐个添加参数，这是最标准的用法
        self.position_mixing['weight'] = nn.Parameter(torch.zeros(head, patch_size ** 2, patch_size ** 2))
        self.position_mixing['bias'] = nn.Parameter(torch.zeros(1, head, 1, patch_size ** 2))

    def forward(self, query, context):
        # x: [N, C, H, W]
        n, c, h, w = context.shape  # [128, 512, 8, 8]
        context = context.reshape(n, c, -1)
        context = rearrange(self.norm(context.transpose(1, -1)), 'b n (h d) -> b h d n', h=self.head)
        context_mlp = torch.einsum('bhdn, hnm -> bhdm', context, self.position_mixing['weight']) + self.position_mixing[
            'bias']
        context = context + context_mlp
        context = context.reshape(n, c, h, w)

        # g_x: [N, HxW, C]
        g_x = self.g(context).view(n, self.inter_channels, -1)
        g_x = rearrange(g_x, 'b (h dim) n -> (b h) dim n', h=self.head)
        g_x = g_x.permute(0, 2, 1)

        # theta_x: [N, HxW, C], phi_x: [N, C, HxW]
        if self.mode == 'gaussian':
            theta_x = query.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(context).view(n, self.in_channels, -1)
            else:
                phi_x = context.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(query).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(context).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(query).view(n, self.inter_channels, -1)
            theta_x = rearrange(theta_x, 'b (h dim) n -> (b h) dim n', h=self.head)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(context).view(n, self.inter_channels, -1)
            phi_x = rearrange(phi_x, 'b (h dim) n -> (b h) dim n', h=self.head)

        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        y = rearrange(y, '(b h) n dim -> b n (h dim)', h=self.head)
        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                                    *query.size()[2:])

        output = query + self.conv_out(y)
        return output


class BSHead(BaseDecodeHead):
    def __init__(self, in_channels=[256, 512, 1024, 2048], in_index=[0, 1, 2, 3], num_classes=2, channels=512):
        super().__init__(input_transform='multiple_select',
                         in_channels=in_channels, in_index=[0, 1, 2, 3], num_classes=num_classes,
                         channels=512, dropout_ratio=0.1, norm_cfg=norm_cfg, align_corners=False)
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)
        #新增c1
        self.unify_c1 = ConvModule(self.in_channels[0], self.channels, 1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        
        self.unify_c2 = ConvModule(self.in_channels[1], self.channels, 1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.unify_c3 = ConvModule(self.in_channels[2], self.channels, 1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.unify_c4 = ConvModule(self.in_channels[3], self.channels, 1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        
        # 实例化3D注意力模块
        self.d3d_attention = Decomposed3DAttention(channels=self.channels)

        self.squeeze = ConvModule(  # stage2 3 4
            sum(self.in_channels) - self.in_channels[0],
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.channels = channels
        self.in_index = in_index
        self.conv00 = nn.Conv2d(self.in_channels[-1], self.channels, 1)
        self.conv0 = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=5,
            padding=4,
            dilation=2,
            # groups=self.channels,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.conv_spatial = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=7,
            padding=9,
            dilation=3,
            # groups=self.channels,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.conv11 = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,  # // 2
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.conv12 = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,  # // 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        # ============================
        #  Context Attention Module  #
        # ============================
        self.patch_size = 4
        self.conta = ContAttn(
            in_channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode='embedded_gaussian',
            head=16,
            patch_size=self.patch_size)

        self.conv13 = ConvModule(
            in_channels=self.channels * 2,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.attn_fuse = ConvModule(
            in_channels=self.channels * 2,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.low_level_fuse = ConvModule(
            in_channels=self.channels + self.in_channels[0],
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = nn.Conv2d(in_channels, self.channels, 1)
            self.lateral_convs.append(l_conv)

    def get_patch(self, x, patch_size):
        n, _, h, w = x.shape
        # get patch
        patch = F.unfold(x, kernel_size=patch_size, stride=patch_size)
        patch = rearrange(patch, 'b (c ph pw) (nh nw) -> (b nh nw) c ph pw',
                          ph=patch_size, pw=patch_size,
                          nh=h // patch_size, nw=w // patch_size)
        return patch

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        c1, c2, c3, c4 = inputs

        # # --- 新的下采样融合流程 ---

        # # 1. 确定目标尺寸为 c4 的尺寸
        # target_size = c4.shape[2:] # 例如 (16, 16)

        # # 2. 统一通道数
        # c2_unified = self.unify_c2(c2)
        # c3_unified = self.unify_c3(c3)
        # c4_unified = self.unify_c4(c4)

        # # 3. 将 c2 和 c3 下采样到目标尺寸
        # c2_resized = resize(c2_unified, size=target_size, mode='bilinear', align_corners=self.align_corners)
        # c3_resized = resize(c3_unified, size=target_size, mode='bilinear', align_corners=self.align_corners)

        # # 4. 将三个特征图堆叠起来，创建 "深度" 维度
        # # (B, C, H, W) -> (B, C, 1, H, W)
        # c2_stacked = c2_resized.unsqueeze(2)
        # c3_stacked = c3_resized.unsqueeze(2)
        # c4_stacked = c4_unified.unsqueeze(2) # 注意这里用的是 c4_unified，因为它已经是目标尺寸了

        # # 拼接成 (B, C, D, H, W)，这里的 D=3, H和W都变小了
        # multi_level_features = torch.cat([c2_stacked, c3_stacked, c4_stacked], dim=2)
        
        # # 5. 应用3D注意力
        # attended_features = self.d3d_attention(multi_level_features) # 输出仍为 (B, C, 3, H, W)

        # # 6. 融合深度维度，得到最终的2D特征图
        # x_attn = attended_features.mean(dim=2) # (B, C, H, W)

        # # 确保返回给 EdgeHead 的 out_semantic 是4D的
        # out_semantic = x_attn 

        # 上采样融合流程
        # c2_unified = self.unify_c2(c2)
        # c3_unified = self.unify_c3(c3)
        # c4_unified = self.unify_c4(c4)

        # # 2. 将它们上采样到相同的尺寸（例如 c2 的尺寸）
        # target_size = c2_unified.shape[2:]
        # c3_resized = resize(c3_unified, size=target_size, mode='bilinear', align_corners=self.align_corners)
        # c4_resized = resize(c4_unified, size=target_size, mode='bilinear', align_corners=self.align_corners)

        # # 3. 将三个特征图堆叠起来，创建 "深度" 维度
        # # (B, C, H, W) -> (B, C, 1, H, W)
        # c2_stacked = c2_unified.unsqueeze(2)
        # c3_stacked = c3_resized.unsqueeze(2)
        # c4_stacked = c4_resized.unsqueeze(2)
        
        # # 拼接成 (B, C, D, H, W)，这里的 D=3
        # multi_level_features = torch.cat([c2_stacked, c3_stacked, c4_stacked], dim=2)

        target_size = c4.shape[2:]

        c1_unified = self.unify_c1(c1)
        c2_unified = self.unify_c2(c2)
        c3_unified = self.unify_c3(c3)
        c4_unified = self.unify_c4(c4)

        c1_resized = resize(c1_unified, size=target_size, mode='bilinear', align_corners=self.align_corners)
        c2_resized = resize(c2_unified, size=target_size, mode='bilinear', align_corners=self.align_corners)
        c3_resized = resize(c3_unified, size=target_size, mode='bilinear', align_corners=self.align_corners)

        c1_stacked = c1_resized.unsqueeze(2)
        c2_stacked = c2_resized.unsqueeze(2)
        c3_stacked = c3_resized.unsqueeze(2)
        c4_stacked = c4_unified.unsqueeze(2)

        # D 变为 4
        multi_level_features = torch.cat([c1_stacked, c2_stacked, c3_stacked, c4_stacked], dim=2)

        
        
        # 4. 应用3D注意力
        attended_features = self.d3d_attention(multi_level_features) # 输出仍为 (B, C, 3, H, W)
        #attended_features = multi_level_features   #消融注意力实验

        # 5. 融合深度维度，得到最终的2D特征图
        # 我们可以简单地取平均或使用一个1x1卷积来融合
        x_attn = attended_features.mean(dim=2) # 简单取平均，得到 (B, C, H, W)

        # 现在，x_attn 就是一个融合了多尺度上下文信息的强大特征图
        # 你可以用它来替换你原来代码中的 x_fuse 或 x_attn
        out_semantic = x_attn 

        # 接下来可以继续你原来的FPN结构或者直接与 c1 融合
        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(x_attn) # 用新的 x_attn 替换原来FPN的顶层特征
        


        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)
        out = self.cls_seg(laterals[0])
        out_ori = laterals[0]
        return out, out_semantic, out_ori
