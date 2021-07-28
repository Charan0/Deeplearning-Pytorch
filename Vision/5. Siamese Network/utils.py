import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Union


def train_fn(network: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, loader: DataLoader,
             device: Union[str, torch.device]):
    loop = tqdm(enumerate(loader), leave=True, total=len(loader))  # A tqdm loop
    network.train()  # Network on train mode
    running_loss = 0.0
    for batch_idx, (anchors, positives) in loop:
        anchors = anchors.to(device)
        positives = positives.to(device)

        # Get the encoded representation
        anc_embeddings = network(anchors)
        pos_embeddings = network(positives)

        optimizer.zero_grad()

        loss = criterion(anc_embeddings, pos_embeddings)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        loop.set_description(f'Step: [{batch_idx + 1}/{len(loader)}]')
        loop.set_postfix(loss=running_loss / (batch_idx + 1))

    return running_loss / len(loader)


def cosine_similarity(network: torch.nn.Module, sample1: torch.Tensor, sample2: torch.Tensor):
    def normalize(x: torch.Tensor):
        return x / torch.norm(x)
    network.eval()  # Network on eval mode
    emb1, emb2 = network(sample1).squeeze(), network(sample2).squeeze()
    assert emb1.ndim == 1 and emb2.ndim == 1
    emb1, emb2 = normalize(emb1), normalize(emb2)
    # Returns the cosine similarity
    return torch.dot(emb1, emb2.T)  # -1 <= sim <= 0 => Not similar; 0 < sim <= 1  => similar
