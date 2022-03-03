import copy

import cv2
from PIL import Image
from torch import nn
from torchvision import models


import PIL
import torch
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm

from utils.misc.input import RealSense
from utils.model.wrappers import Segmentator


if __name__ == '__main__':
    model = models.segmentation.fcn_resnet101(pretrained=False)
    model.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
    model.load_state_dict(torch.load('./checkpoints/sym4/latest'), strict=False)
    model.eval()

    model = Segmentator(model, device='cuda')

    vid = cv2.VideoCapture(0)
    # camera = RealSense()

    with tqdm(total=10000) as pbar:
        while (True):
            ret, frame = vid.read()
            # frame, _ = camera.read()

            # frame = np.array(Image.open('test_rgb.png'))
            # depth = np.array(Image.open('test_depth.png')).astype(np.uint16)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            segmented, categories = model(frame)

            overlay = copy.deepcopy(frame)
            if np.any(categories == 1):
                overlay[categories == 1] = np.array([0, 0, 128])
            res = cv2.addWeighted(frame, 1, overlay, 0.5, 0)

            cv2.imshow('frame', res)
            cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            pbar.update(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    vid.release()
    cv2.destroyAllWindows()