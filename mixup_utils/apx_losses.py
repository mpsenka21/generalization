import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import (mixup, full_mixup, CrossEntropyLoss)

# Module to compute all losses that approximate mixup training. These include:

#  - Double sum (eq. 4)
#  - Standard Taylor, global order 2 (eq. 12)
#  - Our Taylor, up to order 2 for each \epsilon and \delta (eq. 4)
#  --- Note that for cross entropy loss, this only amounts to adding an 
#          \epsilon\delta^2 term

# (equation refs are taken from "On Mixup Regularization"): 
# https://arxiv.org/pdf/2006.06049.pdf

# given a pytorch function g (twice differentiable),
# compute matrix-vector products of the form (\nabla_{(x : y)}^2 g)v
# x and y are the input variables for g
# x1 and x2 are strings, either 'x' or 'y'
# --- these indicate which variables to take derivatives w.r.t.
# v is vector to get Hessian's action on
def hvp(g, x, y, x1, x2, v):
    # setting up pytorch stuff to prep for backprop
    xvar = Variable(x, requires_grad=True)
    yvar = Variable(y, requires_grad=True)
    vvar = Variable(v, requires_grad=True)

    # choose which variable x1var corresponds to
    x1var = xvar if x1=='x' else yvar
    x2var = xvar if x2=='x' else yvar
    
    score = g(xvar, yvar)
    
    grad, = torch.autograd.grad(score, x1var, create_graph=True)
    # print(grad)
    total = torch.sum(grad * vvar)
    # print(total)
    
    if xvar.grad:
        xvar.grad.data.zero_()
    if yvar.grad:
        yvar.grad.data.zero_()
        
    grad2, = torch.autograd.grad(total, x2var, create_graph=True, allow_unused=True)
    # print(grad2)
    return grad2

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
    if use_gpu:
        miximages = miximages.cuda()
        mixlabels = mixlabels.cuda()

    criterion = CrossEntropyLoss(size_average=True)
    predictions = model(miximages)
    return criterion(predictions, mixlabels)

# double sum loss
# mixup where every pair of images is combined
# fixlam: whether setting lambda to 0.5 everywhere
def doublesum_loss(images, labels, alpha, n_classes, fixlam, model, use_gpu):
    miximages, mixlabels = full_mixup(images, labels, alpha, n_classes, fixlam)
    if use_gpu:
        miximages = miximages.cuda()
        mixlabels = mixlabels.cuda()
        
    criterion = CrossEntropyLoss(size_average=True)
    predictions = model(miximages)
    return criterion(predictions, mixlabels)
