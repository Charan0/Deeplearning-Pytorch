import torch
import torch.nn as nn
from typing import Union


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k_size: Union[int, tuple], **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, k_size, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x: torch.Tensor):
        return self.func(x)


class SiameseNetwork(nn.Module):
    def __init__(self, encoder_network: nn.Module = None, emb_dim: int = 1024,
                 rate: float = 0.5, freeze: bool = False):
        super().__init__()
        self.emb_dim = emb_dim
        self.rate = rate
        if encoder_network is None:
            self.siamese_network = self.get_model()
        else:
            encoder_network.requires_grad_(not freeze)
            pretrained_blocks = list(encoder_network.children())[:-1]  # Remove the top layer
            self.siamese_network = nn.Sequential(
                *pretrained_blocks,  # Output = [2048, 1, 1]
                nn.Conv2d(2048, emb_dim, (1, 1)),
                Lambda(lambda x: x.squeeze())
            )

    def get_model(self):
        return nn.Sequential(
            ConvBlock(3, 32, (3, 3)),
            nn.MaxPool2d((2, 2)),
            ConvBlock(32, 64, (3, 3), stride=(2, 2)),
            ConvBlock(64, 128, (5, 5)),
            nn.MaxPool2d((2, 2)),
            ConvBlock(128, 256, (3, 3), stride=(2, 2)),
            ConvBlock(256, 512, (3, 3)),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(512, self.emb_dim, (1, 1)),  # Change the channel dimensions
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),  # Output shape (N, emb_dim)
            # Fully connected layers
            nn.Linear(self.emb_dim * 3 * 3, self.emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.rate),
            nn.Linear(self.emb_dim, self.emb_dim),
        )

    def forward(self, images: torch.Tensor):
        return self.siamese_network(images)
