from __future__ import annotations

import torch.nn as nn
from torchvision import models


class NotebookMLPHead(nn.Sequential):
    """PyTorch equivalent of the classifier used in the reference notebook."""

    def __init__(self, in_features: int, num_classes: int = 2,
                 hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )


def _make_head(head: str, features: int, dropout: float, hidden_dim: int):
    if head == "simple":
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(features, 2))
    if head == "notebook_mlp":
        return NotebookMLPHead(features, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unsupported classifier head: {head}")


def _freeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def keep_frozen_backbone_in_eval(model: nn.Module) -> None:
    """Prevent BatchNorm/dropout state changes inside a frozen backbone."""
    if not getattr(model, "_crossmed_freeze_backbone", False):
        return
    head_name = model._crossmed_head_name
    for name, child in model.named_children():
        if name != head_name:
            child.eval()


def build_model(architecture="resnet18", pretrained=True, dropout=0.2,
                head="simple", hidden_dim=256, freeze_backbone=False):
    """Build a backbone plus either the baseline or notebook-style head."""
    if architecture in {"resnet18", "resnet50", "resnet101"}:
        constructors = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
            "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
        }
        constructor, default_weights = constructors[architecture]
        model = constructor(weights=default_weights if pretrained else None)
        features = model.fc.in_features
        if freeze_backbone:
            _freeze(model)
        model.fc = _make_head(head, features, dropout, hidden_dim)
        head_name = "fc"
    elif architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        features = model.classifier[-1].in_features
        if freeze_backbone:
            _freeze(model)
        model.classifier = _make_head(head, features, dropout, hidden_dim)
        head_name = "classifier"
    elif architecture == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if freeze_backbone:
            _freeze(model)
        model.classifier = _make_head(head, 512, dropout, hidden_dim)
        head_name = "classifier"
    elif architecture == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        features = model.heads.head.in_features
        if freeze_backbone:
            _freeze(model)
        model.heads = _make_head(head, features, dropout, hidden_dim)
        head_name = "heads"
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    model._crossmed_freeze_backbone = freeze_backbone
    model._crossmed_head_name = head_name
    return model
