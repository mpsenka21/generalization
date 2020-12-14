import torch
from torch.autograd import Variable

# Module to compute and evaluate the Taylor approximate loss in the paper on larger-scale tasks

#  - Standard Taylor, global order 2 (eq. 12)
#  - Our Taylor, up to order 2 for each \epsilon and \delta (eq. 4)
#  --- Note that for cross entropy loss, this only amounts to adding an 
#          \epsilon\delta^2 term

# (equation refs are taken from "On Mixup Regularization"): 
# https://arxiv.org/pdf/2006.06049.pdf

### some modules for computing the necessary covariances, approximately with SVD as needed ###

# takes images and one-hot target vectors as input and computes the means and covariances of the data that are necessary to compute Taylor-approximate loss
def compute_moments(data, targets):
    assert(data.shape[0] == targets.shape[0])
    num = data.shape[0]
    x = data.reshape((num, -1))
    y = targets.reshape((num, -1))
    
    xbar = x.mean(axis=0)
    xcent = x - xbar
    ybar = y.mean(axis=0)
    ycent = y - ybar
    
    xxcov = 1/num * torch.matmul(torch.transpose(xcent, 0, 1), xcent)
    xycov = 1/num * torch.matmul(torch.transpose(xcent, 0, 1), ycent)
    
    return xbar, ybar, xxcov, xycov

# takes a covariance matrix X as input
# and returns U, S, V such that X ~ U * diag(S) * V^T
def decomposition(cov, n_components):
    U, S, V = torch.svd(cov)
    return U[:, :n_components], S[:n_components], V[:, :n_components]

# manual cross_entropy for single model output x (not necessarily distribution)
# and one-hot encoded label y, each a vector-Pytorch tensor

# needed for hvp (see below)

# X and Y are (N x x/y_dim) matrices, batch size N
def cross_entropy_manual(X, Y):
    # note X.shape[0] is the batch size
    X_softmax = X.exp() / X.exp().sum(axis=1).reshape((X.shape[0], 1))
    # TODO: check pytorch uses base 2
    return -(Y * torch.log2(X_softmax)).sum()

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
    # setting up pytorch stuff to prepare for backprop
    vvar = Variable(v, requires_grad=True)

    # extract batch size
    N = X.shape[0]
    
    Xvar = Variable(X, requires_grad=True)
    Yvar = Variable(Y, requires_grad=True)
    model_eval = model(Xvar.reshape(data_shape))
    
    # xvar = Variable(X[i,:], requires_grad=True)
    # yvar = Variable(Y[i,:], requires_grad=True)
    # choose which variable x1var corresponds to
    x1var = Xvar if x1=='x' else Yvar
    x2var = Xvar if x2=='x' else Yvar
    
    score = loss(model_eval, Yvar)

    # gradient w.r.t. entire batch 
    grad, = torch.autograd.grad(score, x1var, create_graph=True)
    # sum over batch elements (avg. at end)
    total = torch.sum(grad.sum(axis=0) * vvar)
    
    if Xvar.grad:
        Xvar.grad.data.zero_()
    if Yvar.grad:
        Yvar.grad.data.zero_()
    
    grad2, = torch.autograd.grad(total, x2var, create_graph=True, allow_unused=True)
    # sum over rows (different elements in batch)
    hvprod = (1/N)*grad2.sum(axis=0)

    return hvprod