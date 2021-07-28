import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import vgg19
import torchvision.transforms as T
from model import SiameseNetwork
from data import SiameseDataset
from utils import train_fn, cosine_similarity

emb_dim = 1024
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor()
])
pretrained_net = vgg19(pretrained=True)

bs = 16
siamese_ds = SiameseDataset("/home/charan/Downloads/sketches", "/home/charan/Downloads/photos", transform=transform)
siamese_dl = DataLoader(siamese_ds, bs, shuffle=True)
