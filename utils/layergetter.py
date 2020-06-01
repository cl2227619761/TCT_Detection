#!/usr/bin/env python
# coding=utf-8
"""
获取特征提取网络的中间层特征
"""
import torch.nn as nn
from torch.jit.annotations import Dict

from collections import OrderedDict


class IntermediateLayerGetter(nn.ModuleDict):
    """
    作用:
        提取模型的中间层特征
    参数:
        model: 卷积神经网络模型
        return_layers: 字典，键为要提取的模块的名称；值为提取到的特征
               新名字
    返回:
        有序字典: 键为新名字；值为提取到的特征张量
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str]
    }

    def __init__(self, model, return_layers):
        # 用来得到新名字
        ori_return_layers = return_layers.copy()
        # 把要提取的模块存进字典里，构建ModuleDict使用
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        # 继承ModuleDict的时候，提供layers字典作为模块
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = ori_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


if __name__ == "__main__":
    from torchvision import models
    import torch
    from torchvision.models._utils import IntermediateLayerGetter as Getter

    x = torch.randn(1, 3, 224, 224)
    model = models.resnet50(pretrained=True)
    return_layers1 = {"layer1": "feat1", "layer2": "feat2"}
    return_layers2 = {"layer1": "feat1", "layer2": "feat2"}

    layergetter = IntermediateLayerGetter(model, return_layers1)
    out = layergetter(x)
    getter = Getter(model, return_layers2)
    out2 = getter(x)
    print((out["feat1"]==out2["feat1"]).all())
    print((out["feat2"]==out2["feat2"]).all())

