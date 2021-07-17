import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: tuple = (64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()  # All the downsampling layers go here
        self.ups = nn.ModuleList()  # All the upsampling layers go here
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling layers
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Upsampling layers
        for feature in reversed(features):
            # features*2 because of skip connections, which are concatenated to the channel dimension
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))  # Makes the image size double
            self.ups.append(DoubleConv(feature*2, feature))  # Followed by two conv

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_block = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x: torch.Tensor):
        skip_connections = []
        # Downsampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)  # A double conv and pool

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upsampling
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_conn = skip_connections[idx // 2]

            if x.shape != skip_conn.shape:
                x = TF.resize(x, size=skip_conn.shape[2:])

            concat_skip = torch.cat((skip_conn, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_block(x)


if __name__ == '__main__':
    inputs = torch.randn(3, 1, 384, 384)
    model = UNet(1, 1)
    outputs = model(inputs)
    assert inputs.shape == outputs.shape
    print(outputs.shape)
    print(inputs.shape)