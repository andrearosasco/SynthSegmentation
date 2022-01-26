import random
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
import wandb
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex, Precision, F1Score, Recall, MeanMetric
from torchvision.transforms import InterpolationMode

from config import Config
from utils.data.testset import TestSet
from utils.data.trainset import TrainSet
from torchvision import models

from functools import reduce

import utils.model.wrappers


def train(train_loader, criterion, optim, model, metrics):
    model.train()
    for i, (img_batch, lbl_batch) in enumerate(train_loader):
        img_batch, lbl_batch = img_batch.cuda(), lbl_batch.cuda()
        logits = model(img_batch)['out']

        loss = criterion(logits.reshape(logits.shape[0], logits.shape[1], -1),
                         lbl_batch.reshape(lbl_batch.shape[0], -1))

        if i % 10 == 0:
            reduce(lambda k, v: wandb.log({f'train/{k}': v(torch.argmax(logits, dim=1), lbl_batch),
                                           'train/loss': loss.item()}), metrics.items())

        optim.zero_grad()
        loss.backward()
        optim.step()


@torch.no_grad()
def valid(valid_loader, model, metrics):
    model.eval()
    for i, (img_batch, lbl_batch) in enumerate(valid_loader):
        img_batch, lbl_batch = img_batch.cuda(), lbl_batch.cuda()
        logits = model(img_batch)['out']

        for mtr in metrics.values():
            mtr(torch.argmax(logits, dim=1),
                lbl_batch)  # TODO check softmax dim and check if metric is averaging correctly when no class

    return metrics


def run():
    epoch = Config.TrainConfig.epoch

    model = models.segmentation.fcn_resnet101(pretrained=True).train()
    model.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
    model.cuda()

    if Config.EvalConfig.wandb:
        wandb.init(project='segmentation', config=Config.to_dict())
        wandb.watch(model, log='all')

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 0.7]).cuda())  # TODO change weights

    optim = SGD(params=model.parameters(), lr=Config.TrainConfig.lr, momentum=Config.TrainConfig.momentum,
                weight_decay=Config.TrainConfig.weight_decay)

    dataset = TrainSet(transform=T.Compose([T.Resize((256, 192), InterpolationMode.BILINEAR),
                                              T.ToTensor(),
                                              T.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])]),
                         target_transform=T.Compose([T.Resize((256, 192), InterpolationMode.NEAREST),
                                                     lambda x: torch.tensor(np.array(x), dtype=torch.long)]))

    train_set, valid_set_unity = torch.utils.data.random_split(dataset, [7891, 200])

    train_loader = DataLoader(train_set, batch_size=Config.DataConfig.Train.mb_size, shuffle=True,
                              num_workers=Config.DataConfig.num_worker)

    valid_set_ycb = TestSet(splits=Path('./data'), paths=Config.DataConfig.Eval.paths,
                        transform=T.Compose([T.Resize((256, 192), InterpolationMode.BILINEAR),
                                             T.ToTensor(),
                                             T.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])]),
                        target_transform=T.Compose([T.Resize((256, 192), InterpolationMode.NEAREST),
                                                    lambda x: torch.tensor(np.array(x), dtype=torch.long)]))

    valid_loader_ycb = DataLoader(valid_set_ycb, batch_size=Config.DataConfig.Eval.mb_size, shuffle=False,
                              num_workers=Config.DataConfig.num_worker)

    valid_loader_unity = DataLoader(valid_set_unity, batch_size=Config.DataConfig.Eval.mb_size, shuffle=False,
                              num_workers=Config.DataConfig.num_worker)

    metrics = {
        'jaccard': JaccardIndex(num_classes=2, threshold=0.5, reduction='none').to('cuda'), # TODO
        'precision': Precision(threshold=0.5, multiclass=False).to('cuda'),
        'recall': Recall(threshold=0.5, multiclass=False).to('cuda'),
        'f1score': F1Score(threshold=0.5, multiclass=False).to('cuda')}
    avg_loss = MeanMetric().to('cuda')

    global_step = 0
    best_score = 0
    for e in range(epoch):
        print('Starting epoch ', e)

        with torch.no_grad():
            fixed_img, fixed_gt, sample_img, sample_gt = {}, {}, {}, {}
            print('Validating...')
            model.eval()
            for valid_loader, s in [[valid_loader_ycb, 'ycb'], [valid_loader_unity, 'unity']]:
                sample = random.randint(0, len(valid_loader) - 1)
                for i, (img_batch, lbl_batch) in enumerate(tqdm.tqdm(valid_loader)):
                    img_batch, lbl_batch = img_batch.cuda(), lbl_batch.cuda()
                    logits = model(img_batch)['out']

                    for mtr in metrics.values():
                        mtr(torch.argmax(logits, dim=1),
                            lbl_batch)

                    loss = criterion(logits.reshape(logits.shape[0], logits.shape[1], -1),
                                     lbl_batch.reshape(lbl_batch.shape[0], -1))

                    if i == sample:
                        sample_img[s] = img_batch
                        sample_gt[s] = lbl_batch

                    if i == len(valid_loader) - 1:
                        fixed_img[s] = img_batch
                        fixed_gt[s] = lbl_batch

                if s == 'ycb':
                    score = list(metrics.values())[3].compute()
                    if score > best_score:
                        best_score = score
                        torch.save(model.state_dict(), './seg_model')

                if Config.EvalConfig.wandb:
                    for k, v, in metrics.items():
                        wandb.log({f'valid/{k}_{s}': v.compute(),
                                   'global_step': global_step})
                        v.reset()

                wandb.log({f'valid/loss_{s}': avg_loss(loss),
                           'global_step': global_step})

                avg_loss.reset()

            if Config.EvalConfig.wandb:
                for im, gt, idx, tx in [
                                        [fixed_img['ycb'], fixed_gt['ycb'], 3, 'ycb_fixed'],
                                        [sample_img['ycb'], sample_gt['ycb'],
                                         random.randint(0, sample_img['ycb'].shape[0] - 1), 'ycb_random'],
                                        [fixed_img['unity'], fixed_gt['unity'], 0, 'unity_fixed'],
                                        [sample_img['unity'], sample_gt['unity'],
                                         random.randint(0, sample_img['unity'].shape[0] - 1), 'unity_random'],
                                        [train_set[0][0].unsqueeze(0), train_set[0][1].unsqueeze(0), 0, 'fixed_train'],
                                        [*next(iter(train_loader)), 0, 'random_train']
                ]:
                    segmented, classes = utils.model.wrappers.Segmentator.postprocess(model(img_batch)['out'][idx])

                    tr = T.Compose([
                        lambda x: x.div(1 / torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(x.device)),
                        lambda x: x.sub(-torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(x.device)),
                        T.ToPILImage(),
                        T.Resize((480, 640), InterpolationMode.BILINEAR),
                        lambda x: np.array(x)])

                    wb_image = wandb.Image(tr(im[idx]),
                                           masks={'predictions': {'mask_data': classes,
                                                                  'class_labels': {0: 'background', 1: 'box'}},
                                                  'ground_truth': {'mask_data': cv2.resize(
                                                      gt[idx].cpu().numpy(),
                                                      dsize=(640, 480),
                                                      interpolation=cv2.INTER_NEAREST),
                                                      'class_labels': {0: 'background', 1: 'box'}}})
                    wandb.log({f'media/image_{tx}': wb_image})

        print('Training...')
        model.train()
        for i, (img_batch, lbl_batch) in enumerate(tqdm.tqdm(train_loader)):
            img_batch, lbl_batch = img_batch.cuda(), lbl_batch.cuda()
            logits = model(img_batch)['out']

            loss = criterion(logits.reshape(logits.shape[0], logits.shape[1], -1),
                             lbl_batch.reshape(lbl_batch.shape[0], -1))

            optim.zero_grad()
            loss.backward()
            optim.step()

            if i == len(train_loader) - 1:
                if Config.EvalConfig.wandb:
                    for k, v, in metrics.items():
                        wandb.log({f'train/{k}': v(torch.argmax(logits, dim=1), lbl_batch),
                                   'global_step': global_step})
                        v.reset()

                    wandb.log({'train/loss': loss.item(),
                               'global_step': global_step})

            global_step += i + (e * len(valid_loader))



if __name__ == '__main__':
    run()
