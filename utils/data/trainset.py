from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class TrainSet(Dataset):

    def __init__(self, splits=Path('data/unity/sym2/splits'), mode='train', transform=T.ToTensor(),
                 target_transform=lambda x: torch.tensor(np.array(x))):

        if not isinstance(splits, Path):
            splits = Path(splits)

        with (splits / f'{mode}.txt').open('r') as f:
            self.root = Path(f.readline().strip().split()[1])
            self.examples = list(map(lambda x: x.strip().split(), f.readlines()))

        self.len = len(self.examples)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        item = self.examples[idx]

        if item[0] == 'sym2':
            image = Image.open(self.root / item[1])
            label = Image.open(self.root / item[2])

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return self.len


if __name__ == '__main__':
    train_set = TrainSet()
    train_loader = DataLoader(train_set, batch_size=64)

    for i, (img_batch, lbl_batch) in enumerate(train_loader):
        print(i)