import torch
import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self, emb_dim: int = 1024, rate: float = 0.5):
        super().__init__()
        self.siamese_network = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, (3, 3), stride=(2, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, (5, 5), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(128, 256, (3, 3), stride=(2, 2), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, (3, 3), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(512, emb_dim, (1, 1)),  # Change the channel dimensions
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),  # Output shape (N, emb_dim)
            # Fully connected layers
            nn.Linear(emb_dim * 3 * 3, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(rate),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, images: torch.Tensor):
        return self.siamese_network(images)
