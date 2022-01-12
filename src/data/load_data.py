import torch

from src import _PATH_DATA


class dataset:
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = self.data[idx]
        y = self.target[idx]

        return X, y


def load_data():
    train_images = torch.load(_PATH_DATA + "/processed/train_images.pt")
    train_labels = torch.load(_PATH_DATA + "/processed/train_labels.pt")
    test_images = torch.load(_PATH_DATA + "/processed/test_images.pt")
    test_labels = torch.load(_PATH_DATA + "/processed/test_labels.pt")

    return dataset(train_images, train_labels), dataset(test_images, test_labels)
