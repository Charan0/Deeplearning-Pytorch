import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import vgg19
import torchvision.transforms as T
from model import SiameseNetwork
from loss import ContrastiveLoss
from data import SiameseDataset
from utils import train_fn


class GrayScale:
    def __init__(self):
        pass

    def __call__(self, sample: torch.Tensor):
        channels = sample.shape[0]
        if channels == 1:
            sample = sample.squeeze()
            sample = torch.stack([sample, sample, sample])
        return sample


emb_dim = 1024
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor(),
    GrayScale()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

bs = 16
n_epochs = 5
lr = 1e-3
alpha = 0.25

siamese_ds = SiameseDataset("/home/charan/Downloads/sketches", "/home/charan/Downloads/photos", transform=transform)
siamese_dl = DataLoader(siamese_ds, bs, shuffle=True)

network = SiameseNetwork(emb_dim=emb_dim, rate=0.6)
optimizer = optim.Adam(network.parameters(), lr=lr)
loss_fn = ContrastiveLoss(alpha=alpha, device=device)

losses = []
for _ in range(n_epochs):
    loss = train_fn(network, loss_fn, optimizer, siamese_dl, device)
    losses.append(loss)