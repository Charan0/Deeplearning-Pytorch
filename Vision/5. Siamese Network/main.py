import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Union


def train_fn(network: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, loader: DataLoader,
             device: Union[str, torch.device]):
    loop = tqdm(enumerate(loader), leave=True, total=len(loader))
    network.train()
    running_loss = 0.0
    for batch_idx, (anchors, positives, negatives) in loop:
        anchors = anchors.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)

        anc_embeddings = network(anchors)
        pos_embeddings = network(positives)
        neg_embeddings = network(negatives)

        optimizer.zero_grad()

        loss = criterion(anc_embeddings, pos_embeddings, neg_embeddings)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        loop.set_description(f'Step: [{batch_idx + 1}/{len(loader)}]')
        loop.set_postfix(loss=running_loss / (batch_idx + 1))

    return running_loss / len(loader)
