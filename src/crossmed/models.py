from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(architecture="resnet18", pretrained=True, dropout=0.2):
    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, 2))
    elif architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, 2))
    elif architecture == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        features = model.heads.head.in_features
        model.heads = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, 2))
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return model

