from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class SiameseDataset(Dataset):
    def __init__(self, anchors: str, positives: str, transform=None):
        anc_path = Path(anchors)
        pos_path = Path(positives)
        if not anc_path.exists() or not pos_path.exists():
            print("Paths do not exist")
            exit(-1)
        self.anchors = sorted(image for image in anc_path.iterdir())
        self.positives = sorted([image for image in pos_path.iterdir()])
        assert len(self.anchors) == len(self.positives)
        self.transform = transform

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx: int):
        anchor, positive = self.anchors[idx], self.positives[idx]
        anchor, positive = Image.open(anchor), Image.open(positive)
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
        return anchor, positive
