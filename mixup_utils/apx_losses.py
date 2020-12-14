import torch
import torch.nn as nn
from utils import (mixup, full_mixup, CrossEntropyLoss, AverageMeter)
import torch.utils.data
from torch.utils.data import TensorDataset

# Module to compute losses to evaluate the double sum approximation

### Losses ###

# input: images, labels, models
# output: some loss

# warmup/for a sanity check: true loss
def vanilla_loss(images, labels, model, use_gpu):
    if use_gpu:
        images = images.cuda()
        labels = labels.cuda()

    predictions = model(images)
    criterion = nn.CrossEntropyLoss(size_average=True)
    return criterion(predictions, labels)

# mixup loss
# fixlam: will set lambda to 0.5
# ----
# note: difference between CrossEntropyLoss and nn.CrossEntropyLoss is that
# the nn version takes labels as targets whereas the local one
# takes one-hot vectors as targets, which is important for mixup
def mixup_loss(images, labels, alpha, n_classes, fixlam, model, use_gpu):
    miximages, mixlabels = mixup(images, labels, alpha, n_classes, fixlam)
    return compute_loss(miximages, mixlabels, model, use_gpu)

# double sum loss
# mixup where every pair of images is combined
# fixlam: whether setting lambda to 0.5 everywhere
def doublesum_loss(images, labels, alpha, n_classes, fixlam, model, use_gpu):
    miximages, mixlabels = full_mixup(images, labels, alpha, n_classes, fixlam)
    return compute_loss(miximages, mixlabels, model, use_gpu)

def compute_loss(miximages, mixlabels, model, use_gpu):
    doubleset = TensorDataset(miximages, mixlabels)
    doubleloader = torch.utils.data.DataLoader(doubleset, batch_size=8192, shuffle=False)
    
    criterion = CrossEntropyLoss(size_average=True)
    meter = AverageMeter()
    for imgs, lbls in doubleloader:
        if use_gpu:
            imgs = imgs.cuda()
            lbls = lbls.cuda()

        with torch.no_grad(): # this line is key to preventing CUDA out of memory error
            predictions = model(imgs)

        meter.update(criterion(predictions, lbls), imgs.shape[0])

    return meter.avg