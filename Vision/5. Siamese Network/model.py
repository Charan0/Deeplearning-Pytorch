import torch
import torch.nn as nn
from torchvision.models import vgg19

encoder_network = vgg19(pretrained=True)
inp = torch.randn(1, 3, 256, 256)


class SiameseNetwork(nn.Module):
    def __init__(self, network: nn.Module, emb_dim: int = 1024, rate: float = 0.5, freeze: bool = False):
        super().__init__()
        if freeze:
            for param in network.parameters():
                param.requires_grad = False

        self.encoder_network = nn.Sequential(
            network.features,
            network.avgpool,  # Output of size (N, 512, 7, 7)
            nn.Flatten(),
            # Linear Block - 1
            nn.Linear(512 * 7 * 7, 4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(rate),
            # Linear Block - 2
            nn.Linear(4096, 2048, bias=False),
            nn.ReLU(inplace=True),
            # Embedding Layer
            nn.Linear(2048, emb_dim)
        )

    def forward(self, images: torch.Tensor):
        return self.encoder_network(images)


class SimilarityLoss(nn.Module):
    def __init__(self, alpha: float = 0.2):
        super(SimilarityLoss, self).__init__()
        self.alpha = alpha

    def forward(self, anc_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor):
        positive_dist = torch.norm(anc_emb - pos_emb, dim=1)
        negative_dist = torch.norm(anc_emb - neg_emb, dim=1)
        return torch.maximum(positive_dist - negative_dist + self.alpha, torch.zeros_like(positive_dist)).mean()
