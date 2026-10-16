import torch
import torch.nn as nn
try:
    from .resnet import resnet50_v1b
except:
    from resnet import resnet50_v1b
import torch.nn.functional as F
from models.head import *

class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, nclass, aux, backbone='resnet50', jpu=False, pretrained_base=True, **kwargs):
        super(SegBaseModel, self).__init__()
        dilated = False if jpu else True
        self.aux = aux
        self.nclass = nclass
        if backbone == 'resnet50':
            self.pretrained = resnet50_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)

        # self.jpu = JPU([512, 1024, 2048], width=512, **kwargs) if jpu else None

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        return c1, c2, c3, c4

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred


class BSModel(SegBaseModel):
    def __init__(self, nclass, backbone='resnet50', aux=True, edge_aux=True, pretrained_base=True):
        super(BSModel, self).__init__(nclass, backbone=backbone, aux=aux, pretrained_base=pretrained_base)
        self.head = BSHead(in_channels=[256, 512, 1024, 2048], num_classes=nclass,
                           in_index=[0, 1, 2, 3], channels=512)
        self.aux = aux
        self.edge_aux = edge_aux
        if self.aux:
            # self.aux_head = FCNHead(num_convs=1, in_channels=1024, num_classes=nclass, in_index=2, channels=256)
            self.aux_head = BSFuseHead(in_channels=[128, 512], num_classes=nclass, channels=128)
        if self.edge_aux:
            self.edge_head = EdgeHead(in_channels=[256, 512], in_index=[0, 1], channels=128)


    def forward(self, x):
        size = x.size()[2:]
        c = self.base_forward(x)
        outputs = []
        x, x_semantic, x_ori = self.head(c)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        if isinstance(x, (list, tuple)):
            for out in x:
                out = F.interpolate(out, size,  mode='bilinear', align_corners=True)
                outputs.append(out)
        else: # 1
            x0 = F.interpolate(x, size,  mode='bilinear', align_corners=True)
            outputs.append(x0)

        if self.edge_aux:
            edge, edge_ori = self.edge_head(c, x_semantic)
            edge = F.interpolate(edge, size, mode='bilinear', align_corners=True)
            outputs.append(edge)

        if self.aux:
            auxout = self.aux_head(x_ori, edge_ori)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        # print(len(outputs))
        # print(outputs[0].shape, outputs[1].shape, outputs[2].shape)  #[x0, edge, auxout]
        outputs[1], outputs[2] = outputs[2], outputs[1]
        return outputs   #[x0, auxout, edge]





if __name__ == '__main__':
    from tools.flops_params_fps_count import flops_params_fps
    model = BSModel(nclass=2)
    flops_params_fps(model)





