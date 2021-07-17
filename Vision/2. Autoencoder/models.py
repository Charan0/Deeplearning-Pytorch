import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_features: int = 784,  z_dim: int = 100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),  # (N, 784)

            # Fully Connected - 1
            nn.Linear(in_features, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            # Fully Connected - 2
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            # Fully Connected - 3
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, z_dim)
        )

    def forward(self, images):
        return self.encoder(images)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Decoder(nn.Module):
    def __init__(self, z_dim: int = 100, out_features: int = 784):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, out_features),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)


