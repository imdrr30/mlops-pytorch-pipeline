import torch
from src.model import SimpleModel

def test_forward():
    model = SimpleModel()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)
