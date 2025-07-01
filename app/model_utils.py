"""ResNet‑50 backbone returning 2048‑dim features.
Ensures a **real** child module named ``layer4`` is present so Grad‑CAM can
locate it via ``dict(model.named_modules())["layer4"]``.
"""
from collections import OrderedDict
from functools import lru_cache

import torch
import torch.nn as nn
from torchvision import models


@lru_cache(maxsize=2)
def get_feature_extractor(device: str = "cpu") -> nn.Module:
    """Return a frozen ResNet‑50 up to global‑avg‑pool with explicit names."""

    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    modules = OrderedDict([
        ("conv1", resnet.conv1),
        ("bn1", resnet.bn1),
        ("relu", resnet.relu),
        ("maxpool", resnet.maxpool),
        ("layer1", resnet.layer1),
        ("layer2", resnet.layer2),
        ("layer3", resnet.layer3),
        ("layer4", resnet.layer4),  # ← explicit name Grad‑CAM expects
        ("avgpool", resnet.avgpool),
        ("flatten", nn.Flatten(1)),
    ])

    backbone = nn.Sequential(modules)
    backbone.eval().to(device)
    return backbone