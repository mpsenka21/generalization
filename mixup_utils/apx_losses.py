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

# manual cross_entropy for single model output x (not necessarily distribution)
# and one-hot encoded label y, each a vector-Pytorch tensor

# needed for hvp (see below)

def cross_entropy_manual(x, y):
    x_softmax = nn.Softmax(x)
    # TODO: check pytorch uses base 2
    return -(y * torch.log2(x_softmax)).sum()

# given a pytorch function loss(x_i, y_i) (twice differentiable)
# and a neural network 'model', 
# compute matrix-vector products of the form:
# (\nabla_{x1 x2}^2 loss(model(x), y)) @ v

# X is a tensor of size (N, data_dim), where N is the size of the batch,
# and data_dim is the dimension of the input data (flattened)

# Y is a tensor of size (N, c), where c is the number of classes. Each
# row is a Euclidean basis vector corresponding to the true label

# x1 and x2 are strings, either 'x' or 'y'
# --- these indicate which variables to take derivatives w.r.t.

# v is vector to get Hessian's action on
### v should have same dimension as x1, and must be a row vector

# TODO: deal w/ fact that zero hessian returns None object
def hvp(loss, model, data_shape, X, Y, x1, x2, v):
    # setting up pytorch stuff to prep for backprop
    vvar = Variable(v, requires_grad=True)

    # extract batch size
    N = X.shape[0]
    # extract final product shape
    prod_shape = X[0].shape if x2=='x' else Y[0].shape
    hvprod = torch.zeros(prod_shape)
    
    Xvar = Variable(X, requires_grad=True)
    Yvar = Variable(Y, requires_grad=True)
    model_eval = model(Xvar.reshape(data_shape))
    for i in range(N):
        # xvar = Variable(X[i,:], requires_grad=True)
        # yvar = Variable(Y[i,:], requires_grad=True)
        # choose which variable x1var corresponds to
        x1var = Xvar[i,:] if x1=='x' else Yvar[i,:]
        x2var = Xvar[i,:] if x2=='x' else Yvar[i,:]
        
        score = loss(model_eval[i,:], Yvar[i,:])
        
        grad, = torch.autograd.grad(score, x1var, create_graph=True)
        total = torch.sum(grad * vvar)
        
        if xvar.grad:
            xvar.grad.data.zero_()
        if yvar.grad:
            yvar.grad.data.zero_()
        
        hvprod, = hvprod + torch.autograd.grad(total, x2var, create_graph=True, allow_unused=True)

    return (1/N) * hvprod

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
    return criterion(predictions, mixlabels)#, save_path='mixup.png')

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
    return criterion(predictions, mixlabels)#, save_path='doublesum.png')
