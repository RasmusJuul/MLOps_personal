from torch import Size, randn

from src.models.model import CNN


def test_output_shape():
    model = CNN(1, 28, 28, 0)
    x = randn(1, 1, 28, 28)
    assert model(x).data.shape == Size([1, 10])
