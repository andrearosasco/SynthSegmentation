from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class TrainSet(Dataset):

    def __init__(self, root=Path('data/unity'), transform=None, target_transform=None):
        self.rgbs = root / 'RGB'
        self.labels = root / 'SemanticSegmentation'

        self.len = len(list(self.rgbs.glob('*.png')))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        image = Image.open(self.rgbs / f'rgb_{item:06}.png')
        # image.save('test_pil.png')
        label = Image.open(self.labels / f'segmentation_{item:06}.png')

        # image = cv2.imread(str(self.rgbs / f'rgb_{item:06}.png'))
        # cv2.imwrite('test_cv.png', np.array(image))
        # cv2.imshow('', np.array(image))
        # cv2.waitKey(0)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return self.len


if __name__ == '__main__':
    train_set = TrainSet()
    train_loader = DataLoader(train_set, batch_size=64,)

    for i, (img_batch, lbl_batch) in enumerate(train_loader):
        print(i)