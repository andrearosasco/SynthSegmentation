from pathlib import Path

from PIL import Image
import cv2
import scipy.io as scio
import numpy as np
import os
import sys

box_cls = {"003_cracker_box": 2,
           "004_sugar_box": 3,
           "008_pudding_box": 7,
           "009_gelatin_box": 8,
           "036_wood_block": 16,
           "061_foam_brick": 21}


def ycb_preprocess():
    root = Path('../DenseFusion/datasets/ycb')
    data = root / 'YCB_Video_Dataset' / 'data'

    for video in data.glob('*'):
        for frame in video.glob('*'):
            if frame.suffix == '.mat':
                meta = scio.loadmat(frame)
                objs_cls = meta['cls_indexes'].flatten().astype(np.int32)
                if not set(objs_cls).isdisjoint(box_cls.values()):
                    idx = int(frame.stem[:-5])
                    label = cv2.imread(f'{frame.parent}/{idx:06}-label.png')
                    color = cv2.imread(f'{frame.parent}/{idx:06}-color.png')

                    label[~np.isin(label, list(box_cls.values()))] = 0


def unity_preprocess():
    root = Path('data/unity')
    labels = root / 'SemanticSegmentation'
    rgbs = root / 'RGB'

    for f, l in zip(sorted(rgbs.glob('*.png'), key=lambda x: int(str(x.stem)[4:])),
                    sorted(labels.glob('*.png'), key=lambda x: int(str(x.stem)[13:]))):
        rgb = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        label = cv2.imread(str(l))

        label[label != 0] = 1
        label = label[..., 0]

        Image.fromarray(label).save(f'{l.parent}/segmentation_{int(str(l.stem)[13:]):06}.png')
        Image.fromarray(rgb).save(f'{f.parent}/rgb_{int(str(f.stem)[4:]):06}.png')

        l.unlink()
        f.unlink()

if __name__ == '__main__':
    unity_preprocess()
