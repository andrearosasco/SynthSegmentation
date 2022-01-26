import cv2
from PIL import Image
from torch import nn
from torchvision import models


import PIL
import torch
import torchvision.transforms as T
import numpy as np

from utils.model.wrappers import Segmentator


if __name__ == '__main__':
    model = models.segmentation.fcn_resnet101(pretrained=False)
    model.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
    model.load_state_dict(torch.load('./seg_model'), strict=False)
    model.eval()

    model = Segmentator(model, device='cuda')

    vid = cv2.VideoCapture(0)

    while (True):
        ret, frame = vid.read()
        segmented, categories = model(frame)

        res = cv2.addWeighted(frame, 0.4, segmented, 0.3, 0)
        cv2.imshow('frame', res)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()