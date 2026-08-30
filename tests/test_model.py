import torch
from src.model import SimpleModel, _HAS_TORCHVISION, get_model

def test_forward():
    model = SimpleModel()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)


def test_resnet18_forward():
    if not _HAS_TORCHVISION:
        return
    model = get_model(architecture="resnet18", num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)
