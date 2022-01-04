from torch import nn
from torch import unsqueeze
from torch.nn.modules.conv import Conv2d


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduce image size by half  28x28 -> 14x14
            nn.Conv2d(4, 8, 3, 1, 1),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduce image size by half  14x14 -> 7x7
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.Dropout(p=0.2),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 16, 256),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):  # [batch_size,28,28]
        x = unsqueeze(
            x, dim=1
        )  # adds and extra dimension [batch_size,28,28] -> [batch_size,1,28,28]
        x = self.conv(x)
        x = x.view(
            x.size(0), -1
        )  # reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        out = self.fc(x)
        return out
