import torch
import torch.nn as nn
import torch.nn.functional as F
# from .layers import Convx2, DownBlock, UpBlock, WithSE, PreactConvx2
import math


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Convx2(nn.Module):
    def __init__(self, c_in, c_out, bn, padding_mode='zeros'):
        super().__init__()
        conv_args = dict(padding=1, padding_mode=padding_mode, bias=not bn)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, **conv_args)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, **conv_args)
        if bn:
            self.bn1 = nn.BatchNorm2d(c_out)
            self.bn2 = nn.BatchNorm2d(c_out)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class DenseBlock(nn.Module):
    def __init__(self, c_in, c_out, bn, dense_size=8):
        super().__init__()
        conv_args = dict(kernel_size=3, padding=1, bias=not bn)
        self.dense_convs = nn.ModuleList([
            nn.Conv2d(c_in + i * dense_size, dense_size, **conv_args)
            for i in range(4)
        ])
        self.final = nn.Conv2d(c_in + 4 * dense_size, c_out, **conv_args)

        if bn:
            self.bns = nn.ModuleList([
                nn.BatchNorm2d(dense_size)
                for i in range(4)
            ])
            self.bn_final = nn.BatchNorm2d(c_out)
        else:
            self.bns = nn.ModuleList([Identity() for i in range(4)])
            self.bn_final = Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for conv, bn in zip(self.dense_convs, self.bns):
            x = torch.cat([x, self.relu(bn(conv(x)))], dim=1)
        x = self.relu(self.bn_final(self.final(x)))
        return x


class SqueezeExcitation(nn.Module):
    """
    adaptively recalibrates channel-wise feature responses by explicitly
    modelling interdependencies between channels.
    See: https://arxiv.org/abs/1709.01507
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced = int(math.ceil(channels / reduction))
        self.squeeze = nn.Conv2d(channels, reduced, 1)
        self.excite = nn.Conv2d(reduced, channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = F.avg_pool2d(x, x.shape[2:])
        y = self.relu(self.squeeze(y))
        y = torch.sigmoid(self.excite(y))
        return x * y


def WithSE(conv_block, reduction=8):
    def make_block(c_in, c_out, **kwargs):
        return nn.Sequential(
            conv_block(c_in, c_out, **kwargs),
            SqueezeExcitation(c_out, reduction=reduction)
        )
    make_block.__name__ = f"WithSE({conv_block.__name__})"
    return make_block


class DownBlock(nn.Module):
    """
    UNet Downsampling Block
    """
    def __init__(self, c_in, c_out, conv_block=Convx2,
                 bn=True, padding_mode='zeros'):
        super().__init__()
        bias = not bn
        self.convdown = nn.Conv2d(c_in, c_in, 2, stride=2, bias=bias)
        if bn:
            self.bn = nn.BatchNorm2d(c_in)
        else:
            self.bn = Identity()
        self.relu = nn.ReLU(inplace=True)

        self.conv_block = conv_block(c_in, c_out, bn=bn, padding_mode=padding_mode)

    def forward(self, x):
        x = self.relu(self.bn(self.convdown(x)))
        x = self.conv_block(x)
        return x


class UpBlock(nn.Module):
    """
    UNet Upsampling Block
    """
    def __init__(self, c_in, c_out, conv_block=Convx2,
                 bn=True, padding_mode='zeros'):
        super().__init__()
        bias = not bn
        self.up = nn.ConvTranspose2d(c_in, c_in // 2, 2, stride=2, bias=bias)
        if bn:
            self.bn = nn.BatchNorm2d(c_in // 2)
        else:
            self.bn = Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv_block = conv_block(c_in, c_out, bn=bn, padding_mode=padding_mode)

    def forward(self, x, skip):
        x = self.relu(self.bn(self.up(x)))
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x

class HEDUNet(nn.Module):
    """
    A straight-forward HED-UNet implementation
    """

    def __init__(self, input_channels, output_channels=2, base_channels=16,
                 conv_block=Convx2, padding_mode='replicate', batch_norm=False,
                 squeeze_excitation=False, merging='attention', stack_height=5,
                 deep_supervision=True):
        super().__init__()
        bc = base_channels
        if squeeze_excitation:
            conv_block = WithSE(conv_block)
        self.init = nn.Conv2d(input_channels, bc, 1)

        self.output_channels = output_channels

        conv_args = dict(
            conv_block=conv_block,
            bn=batch_norm,
            padding_mode=padding_mode
        )

        self.down_blocks = nn.ModuleList([
            DownBlock((1<<i)*bc, (2<<i)*bc, **conv_args)
            for i in range(stack_height)
        ])

        self.up_blocks = nn.ModuleList([
            UpBlock((2<<i)*bc, (1<<i)*bc, **conv_args)
            for i in reversed(range(stack_height))
        ])

        self.predictors = nn.ModuleList([
            nn.Conv2d((1<<i)*bc, output_channels, 1)
            for i in reversed(range(stack_height + 1))
        ])

        self.deep_supervision = deep_supervision
        self.merging = merging
        if merging == 'attention':
            self.queries = nn.ModuleList([
                nn.Conv2d((1<<i)*bc, output_channels, 1)
                for i in reversed(range(stack_height + 1))
            ])
        elif merging == 'learned':
            self.merge_predictions = nn.Conv2d(output_channels*(stack_height+1), output_channels, 1)
        else:
            # no merging
            pass


    def forward(self, x):
        B, _, H, W = x.shape
        x = self.init(x)

        skip_connections = []
        for block in self.down_blocks:
            skip_connections.append(x)
            x = block(x)

        multilevel_features = [x]
        for block, skip in zip(self.up_blocks, reversed(skip_connections)):
            x = block(x, skip)
            multilevel_features.append(x)

        predictions_list = []
        full_scale_preds = []
        for feature_map, predictor in zip(multilevel_features, self.predictors):
            prediction = predictor(feature_map)
            predictions_list.append(prediction)
            full_scale_preds.append(F.interpolate(prediction, size=(H, W), mode='bilinear', align_corners=True))

        predictions = torch.cat(full_scale_preds, dim=1)

        if self.merging == 'attention':
            queries = [F.interpolate(q(feat), size=(H, W), mode='bilinear', align_corners=True)
                    for q, feat in zip(self.queries, multilevel_features)]
            queries = torch.cat(queries, dim=1)
            queries = queries.reshape(B, -1, self.output_channels, H, W)
            attn = F.softmax(queries, dim=1)
            predictions = predictions.reshape(B, -1, self.output_channels, H, W)
            combined_prediction = torch.sum(attn * predictions, dim=1)
        elif self.merging == 'learned':
            combined_prediction = self.merge_predictions(predictions)
        else:
            combined_prediction = predictions_list[-1]

        if self.deep_supervision:
            return combined_prediction, list(reversed(predictions_list))
        else:
            return combined_prediction

if __name__ == '__main__':
    # 设置模型参数
    input_channels = 3  # 假设输入图像有3个通道（例如RGB图像）
    output_channels = 2  # 假设我们进行二分类
    base_channels = 16
    stack_height = 5
    deep_supervision = False

    # 创建模型实例
    model = HEDUNet(input_channels=3, output_channels=2, base_channels=16,
                    merging='attention', stack_height=5,
                    deep_supervision=False)

    # 创建一个随机的输入张量，形状为(B, C, H, W)
    # 这里我们使用一个随机的4x3x256x256张量作为示例
    B, C, H, W = 4, 3, 512, 512
    input_data = torch.rand(B, C, H, W)

    output = model(input_data)

    # 输出模型的输出
    print("Output shape:", output.shape)