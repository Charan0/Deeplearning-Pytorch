from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np


class SegmentationDataset(Dataset):
    def __init__(self, images_root: str, masks_root: str, transform=None):
        self.images_dir = Path(images_root)
        self.masks_dir = Path(masks_root)
        self.images = [image for image in self.images_dir.iterdir() if image.is_file()]
        self.masks = [mask for mask in self.masks_dir.iterdir() if mask.is_file()]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.images[idx]

        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation['image']
            mask = augmentation['mask']

        return image, mask
