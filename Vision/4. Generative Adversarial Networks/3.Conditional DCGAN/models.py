import torch
from torch.nn.functional import one_hot
from typing import Union
import torch.nn as nn


def conditional_noise(n_samples: int, noise_dim: int, labels: torch.Tensor,
                      n_classes: int = -1, device: Union[str, torch.device] = 'cpu'):
    noise = torch.randn(n_samples, noise_dim, device=device)
    labels = one_hot(labels.to(device), num_classes=n_classes)
    return torch.cat([noise, labels], dim=-1)


def channelize(images: torch.Tensor, labels: torch.Tensor,
               device: Union[str, torch.device] = 'cpu', n_classes: int = -1, image_shape: tuple = (28, 28)):
    images = images.to(device)
    labels = labels.to(device)
    labels = one_hot(labels, num_classes=n_classes)[:, :, None, None]
    channelized_labels = labels.repeat(1, 1, *image_shape)
    channelized_images = torch.cat((images, channelized_labels), dim=1)
    return channelized_images


class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[tuple, int] = 4,
                 stride: Union[tuple, int] = 2, padding: Union[tuple, int] = 1, final_layer: bool = False):
        super(GeneratorBlock, self).__init__()
        if final_layer:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.Tanh()
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, tuple] = 4,
                 stride: Union[int, tuple] = 2, padding: Union[int, tuple] = 1, final_layer: bool = False):
        super(DiscriminatorBlock, self).__init__()
        if final_layer:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(.2)
            )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int = 64):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            Lambda(lambda x: x.view(-1, in_channels, 1, 1)),  # A 1x1 Hypercube
            GeneratorBlock(in_channels, hidden_dim * 6),
            GeneratorBlock(hidden_dim * 6, hidden_dim * 4),
            GeneratorBlock(hidden_dim * 4, hidden_dim * 2),
            GeneratorBlock(hidden_dim * 2, hidden_dim),
            GeneratorBlock(hidden_dim, out_channels, final_layer=True)
        )

    def forward(self, noise):
        return self.generator(noise)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 64):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            DiscriminatorBlock(in_channels, hidden_dim),
            DiscriminatorBlock(hidden_dim, hidden_dim * 2),
            DiscriminatorBlock(hidden_dim * 2, hidden_dim * 4),
            DiscriminatorBlock(hidden_dim * 4, hidden_dim * 6),
            DiscriminatorBlock(hidden_dim * 6, 1, final_layer=True),
        )

    def forward(self, images: torch.Tensor):
        predictions = self.discriminator(images)
        return predictions.view(len(images), -1)


inputs = torch.randn(5, 13, 32, 32)
discriminator = Discriminator(10+3)
outputs = discriminator(inputs)
print(outputs.shape)