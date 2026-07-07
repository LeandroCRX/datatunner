"""
Modelos de Redes Neurais Convolucionais (CNN)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights,
    VGG11_Weights, VGG13_Weights, VGG16_Weights, VGG19_Weights,
    MobileNet_V2_Weights, MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights,
)
from typing import Dict, Any

from datatunner.models.base import BaseModel


RESNET_WEIGHTS = {
    "resnet18": ResNet18_Weights,
    "resnet34": ResNet34_Weights,
    "resnet50": ResNet50_Weights,
    "resnet101": ResNet101_Weights,
}

VGG_WEIGHTS = {
    "vgg11": VGG11_Weights,
    "vgg13": VGG13_Weights,
    "vgg16": VGG16_Weights,
    "vgg19": VGG19_Weights,
}

MOBILENET_WEIGHTS = {
    "mobilenet_v2": MobileNet_V2_Weights,
    "mobilenet_v3_small": MobileNet_V3_Small_Weights,
    "mobilenet_v3_large": MobileNet_V3_Large_Weights,
}


def _get_weights(weights_map, architecture, pretrained):
    if pretrained:
        return weights_map[architecture].DEFAULT
    return None


class ResNetClassifier(BaseModel):
    """Classificador baseado em ResNet"""

    def __init__(
        self,
        num_classes: int,
        architecture: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()

        self.num_classes = num_classes
        self.architecture = architecture
        self.model_name = f"ResNet-{architecture}"

        if architecture not in RESNET_WEIGHTS:
            raise ValueError(f"Arquitetura nao suportada: {architecture}")

        weights = _get_weights(RESNET_WEIGHTS, architecture, pretrained)
        backbone_fn = getattr(models, architecture)
        self.backbone = backbone_fn(weights=weights)

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)

        if freeze_backbone:
            self.freeze_layers()

    def forward(self, x):
        return self.backbone(x)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "architecture": self.architecture,
            "num_classes": self.num_classes,
            "num_parameters": self.count_parameters(),
        }

    def freeze_layers(self, num_layers: int = 0):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True


class VGGClassifier(BaseModel):
    """Classificador baseado em VGG"""

    def __init__(
        self,
        num_classes: int,
        architecture: str = "vgg16",
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()

        self.num_classes = num_classes
        self.architecture = architecture
        self.model_name = f"VGG-{architecture}"

        if architecture not in VGG_WEIGHTS:
            raise ValueError(f"Arquitetura nao suportada: {architecture}")

        weights = _get_weights(VGG_WEIGHTS, architecture, pretrained)
        backbone_fn = getattr(models, architecture)
        self.backbone = backbone_fn(weights=weights)

        num_features = self.backbone.classifier[6].in_features
        self.backbone.classifier[6] = nn.Linear(num_features, num_classes)

        if freeze_backbone:
            self.freeze_layers()

    def forward(self, x):
        return self.backbone(x)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "architecture": self.architecture,
            "num_classes": self.num_classes,
            "num_parameters": self.count_parameters(),
        }


class MobileNetClassifier(BaseModel):
    """Classificador baseado em MobileNet"""

    def __init__(
        self,
        num_classes: int,
        architecture: str = "mobilenet_v2",
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()

        self.num_classes = num_classes
        self.architecture = architecture
        self.model_name = f"MobileNet-{architecture}"

        if architecture not in MOBILENET_WEIGHTS:
            raise ValueError(f"Arquitetura nao suportada: {architecture}")

        weights = _get_weights(MOBILENET_WEIGHTS, architecture, pretrained)
        backbone_fn = getattr(models, architecture)
        self.backbone = backbone_fn(weights=weights)

        if architecture == "mobilenet_v2":
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
        else:
            num_features = self.backbone.classifier[3].in_features
            self.backbone.classifier[3] = nn.Linear(num_features, num_classes)

        if freeze_backbone:
            self.freeze_layers()

    def forward(self, x):
        return self.backbone(x)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "architecture": self.architecture,
            "num_classes": self.num_classes,
            "num_parameters": self.count_parameters(),
        }


class CustomCNN(BaseModel):
    """CNN customizada simples"""
    
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        dropout: float = 0.5
    ):
        """
        Args:
            num_classes: Número de classes
            input_channels: Número de canais de entrada
            dropout: Taxa de dropout
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = "CustomCNN"
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "num_parameters": self.count_parameters()
        }
