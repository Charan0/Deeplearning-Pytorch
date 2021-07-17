import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
from models import Encoder, Decoder
import matplotlib.pyplot as plt

plt.style.use('ggplot')
# %matplotlib inline


def train_fn(encoder: Encoder, decoder: Decoder, optimizer, criterion: nn.Module,
             dataloader: DataLoader, device):
    loop = tqdm(enumerate(dataloader), leave=True, total=len(dataloader))
    avg_loss = 0.0
    for batch_idx, (images, _) in loop:
        images = images.to(device)

        encoded_images = encoder(images)
        reconstructed_images = decoder(encoded_images)
        reconstructed_images = reconstructed_images.view(*images.shape)

        loss = criterion(reconstructed_images, images)
        avg_loss += loss.item()
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        loop.set_description(f'Step: [{batch_idx + 1}/{len(dataloader)}]')
        loop.set_postfix(loss=avg_loss / (batch_idx + 1))

    n_samples = 25
    samples = reconstructed_images.detach().cpu()[:n_samples]
    grid = make_grid(samples, nrow=5)
    return grid, avg_loss / len(dataloader)


train_ds = datasets.CIFAR10('', download=True, train=True, transform=T.ToTensor())

# DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
bs = 64
lr = 1e-3
n_epochs = 100

z_dim = 100
im_flattened = 784

# Models, Optimizer, Loss_Fn
encoder = Encoder(z_dim=z_dim, in_features=im_flattened).to(device)
decoder = Decoder(z_dim=z_dim, out_features=im_flattened).to(device)
optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=lr
)

loss_fn = nn.L1Loss()
dataloader = DataLoader(train_ds, batch_size=bs, shuffle=True)

print(f'Training on Device: {device}')
losses = []
for epoch in range(n_epochs):
    samples_grid, loss = train_fn(encoder, decoder, optimizer, loss_fn, dataloader, device)
    losses.append(loss)
    save_image(samples_grid, f'Samples-{epoch + 1}.png')

plt.plot(losses)
plt.show()
