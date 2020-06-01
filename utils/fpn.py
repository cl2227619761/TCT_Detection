#!/usr/bin/env python
# coding=utf-8
"""
这是特征金字塔模块
"""
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.jit.annotations import Dict, List


from collections import OrderedDict


class FeaturePyramidNetwork(nn.Module):
    """
    特征金字塔模块
    参数:
        in_channels_list: 列表，[C2,C3,C4,C5]输入通道数列表
        out_channel: 经过FPN之后每个水平的输出通道数，都相同
        extra_block: 在P5基础上进行的计算得到P6
    """

    def __init__(self, in_channels_list, out_channel, extra_block=None):
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()  # 存储1x1conv
        self.layer_blocks = nn.ModuleList()  # 存储3x3conv
        for in_channel in in_channels_list:
            inner_block = nn.Conv2d(in_channel, out_channel, 1)
            layer_block = nn.Conv2d(out_channel, out_channel, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        self.extra_block = extra_block

    def get_result_from_inner_blocks(self, x, idx):
        # type: (Tensor, int)
        """
        对C2,C3,C4,C5中的每个分别进行1x1conv
        参数:
            x:{C2:tensor, C3:tensor, C4:tensor, C5:tensor}
            idx:索引
        """
        num_blocks = 0
        for m in self.inner_blocks:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        out = x
        i = 0
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x, idx):
        # type: (Tensor, int)
        """
        执行3x3conv得到P2,P3,P4,P5，做法同上
        """
        num_blocks = 0
        for m in self.layer_blocks:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        out = x
        i = 0
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x):
        # type: (Dict[str, Tensor])
        """
        对FPN模块进行组装
        参数:
            x: {C2:tensor, C3:tensor, C4:tensor, C5:tensor}
        """
        names = list(x.keys())
        x = list(x.values())
        result = []
        # 先把最顶层的C5计算出来，因为它没有+的过程
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        last_layer = self.get_result_from_layer_blocks(last_inner, -1)
        result.append(last_layer)
        for idx in range(len(x)-2, -1, -1):
            inner = self.get_result_from_inner_blocks(x[idx], idx)
            upsample = F.interpolate(last_inner, inner.shape[-2:])
            last_inner = inner + upsample
            layer = self.get_result_from_layer_blocks(last_inner, idx)
            result.insert(0, layer)

        if self.extra_block is not None:
            names, result = self.extra_block(result, names)

        out = OrderedDict([(k, v) for k, v in zip(names, result)])
        return out


class MaxpoolOnP5(nn.Module):
    """
    在P5的基础上进行简单的下采样，得到P6
    """
    
    def forward(self, result, name):
        # type: (List[str], List[Tensor])
        name.append("pool")
        p6 = F.max_pool2d(result[-1], 1, 2, 0)
        result.append(p6)
        return name, result


if __name__ == "__main__":
    import torch
    from layergetter import IntermediateLayerGetter
    from torchvision import models
    from torchvision.models._utils import IntermediateLayerGetter as Getter
    from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork as Feat
    from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


    image = torch.randn(1, 3, 224, 224)
    model = models.resnet50(pretrained=True)

    return_layers = {"layer1": "feat1", "layer2": "feat2",
                     "layer3": "feat3", "layer4": "feat4"}
    return_layers2 = {"layer1": "feat1", "layer2": "feat2",
                     "layer3": "feat3", "layer4": "feat4"}
    body = IntermediateLayerGetter(model, return_layers)
    fpn = FeaturePyramidNetwork([256, 512, 1024, 2048], 256, MaxpoolOnP5())
    x = body(image)
    out = fpn(x)
    body2 = Getter(model, return_layers2)
    fpn2 = Feat([256, 512, 1024, 2048], 256, LastLevelMaxPool())
    x2 = body2(image)
    out2 = fpn2(x2)
    body3 = Getter(model, return_layers2)
    fpn3 = Feat([256, 512, 1024, 2048], 256, LastLevelMaxPool())
    x3 = body3(image)
    out3 = fpn3(x2)
    import ipdb;ipdb.set_trace()


