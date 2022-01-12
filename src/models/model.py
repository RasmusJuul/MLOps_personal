from torch import nn, unsqueeze


class ResNetBlock(nn.Module):
    def __init__(self, features, droprate):
        super(ResNetBlock, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(
                in_channels=features, out_channels=features, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(features),
            nn.Dropout(p=droprate),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.conv_bn(x)
        x = self.activation(x)
        x = self.conv_bn(x)
        x = x + identity
        out = self.activation(x)
        return out


# Define Residual network
class ResNet(nn.Module):
    def __init__(self, features, height, width, droprate, num_blocks=3):
        super(ResNet, self).__init__()
        # First conv layers needs to output the desired number of features.
        conv_layers = [
            nn.Conv2d(1, features, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=droprate),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        ]  # Reduce image size by half  28x28 -> 14x14

        for i in range(num_blocks):
            conv_layers.append(ResNetBlock(features, droprate))

        conv_layers.append(
            nn.Sequential(
                nn.MaxPool2d(2, 2),  # Reduce image size by half 14x14 -> 7x7
                nn.Conv2d(features, 2 * features, kernel_size=3, stride=1, padding=1),
                nn.Dropout(p=droprate),
                nn.ReLU(),
            )
        )

        self.blocks = nn.Sequential(*conv_layers)

        self.fc = nn.Sequential(
            nn.Linear(int(height / 4) * int(width / 4) * 2 * features, 256),
            nn.Dropout(p=droprate),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(p=droprate),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):  # [batch_size,1,28,28]
        x = self.blocks(x)
        # reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


class CNN(nn.Module):
    def __init__(self, features, height, width, droprate):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, features, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=droprate),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduce image size by half  28x28 -> 14x14
            nn.Conv2d(features, features * 2, 3, 1, 1),
            nn.Dropout(p=droprate),
            nn.ReLU(),
            nn.Conv2d(features * 2, features * 2, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout(p=droprate),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduce image size by half  14x14 -> 7x7
            nn.Conv2d(features * 2, features * 4, 3, 1, 1),
            nn.Dropout(p=droprate),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(int(height / 4) * int(width / 4) * 4 * features, 256),
            nn.Dropout(p=droprate),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(p=droprate),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):  # [batch_size,1,28,28]
        x = self.conv(x)
        x = x.view(
            x.size(0), -1
        )  # reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        out = self.fc(x)
        return out
