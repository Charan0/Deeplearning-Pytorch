import torch
import torch.nn as nn


class SiameseNetwork(nn.Module):
    # TODO: Use Resnet or InceptionV3 instead of vgg19 - too many trainable parameters bruh!
    def __init__(self, encoder_network: nn.Module, emb_dim: int = 1024, rate: float = 0.5, freeze: bool = False):
        super().__init__()
        if freeze:
            encoder_network.requires_grad_(False)

        self.siamese_network = nn.Sequential(
            encoder_network.features,
            encoder_network.avgpool,  # Output of size (N, 512, 7, 7)
            # Increases the channel dimensions, reduces spatial dimension
            nn.Conv2d(512, 1024, kernel_size=(5, 5)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),  # Output of size (N, 1024, 1, 1)
            nn.Conv2d(1024, emb_dim, (1, 1)),  # Change the channel dimensions
            nn.Flatten(),  # Output shape (N, emb_dim)
            # Fully connected layers
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(rate),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, images: torch.Tensor):
        return self.siamese_network(images)
