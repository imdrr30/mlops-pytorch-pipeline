import torch
import torch.nn as nn

try:
    from torchvision import models
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False


class SimpleModel(nn.Module):
    """Small CNN used as a fallback and for quick tests."""
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_model(architecture="simple_cnn", num_classes=10, pretrained=False):
    """Return a model for classification. Uses ResNet18 when torchvision is available.

    Falls back to `SimpleModel` when torchvision isn't available or when `pretrained` is False.
    """
    if _HAS_TORCHVISION and architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        # adapt final layer
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
        return model
    return SimpleModel(in_channels=3, num_classes=num_classes)
