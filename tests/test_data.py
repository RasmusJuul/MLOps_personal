from torch import Size
import pytest

from src.data.load_data import load_data

train_dataset, test_dataset = load_data()


def test_length():
    assert len(train_dataset) == 40000
    assert len(test_dataset) == 5000


# assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
# assert that all labels are represented
def test_shape():
    for i in range(len(train_dataset)):
        assert train_dataset.__getitem__(i)[0].shape == Size([1, 28, 28])
        assert train_dataset.__getitem__(i)[1] != None
        assert train_dataset.__getitem__(i)[1] != Size([0])

    for i in range(len(test_dataset)):
        assert test_dataset.__getitem__(i)[0].shape == Size([1, 28, 28])
        assert test_dataset.__getitem__(i)[1] != None
        assert test_dataset.__getitem__(i)[1] != Size([0])