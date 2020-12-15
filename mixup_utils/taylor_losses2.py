import torch
import numpy as np
from torch.autograd import Variable
import time

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
    num_classes = targets.shape[1]
    x = data.reshape((num, -1))
    y = targets.reshape((num, -1))

    x_dim = x.shape[1]
    
    xbar = x.mean(axis=0)
    xcent = x - xbar
    ybar = y.mean(axis=0)
    ycent = y - ybar
    
    xxcov = 1/num * torch.matmul(torch.transpose(xcent, 0, 1), xcent)
    xycov = 1/num * torch.matmul(torch.transpose(xcent, 0, 1), ycent)

    T = torch.zeros((num_classes, x_dim, x_dim)).cuda()
    for i in range(num_classes):
        # xcent is (num by x_dim)
        # T[i,:,:] = E_j(y'_j(c) x'_j x'_j^T)
        T[i,:,:] = (1/num)*xcent.t() @ (xcent * ycent[:,i].reshape((num, 1)))
    
    return xbar, ybar, xxcov, xycov, T

# takes a covariance matrix X as input
# and returns U, S, V such that X ~ U * diag(S) * V^T
def decomposition(cov, n_components):
    U, S, V = torch.svd(cov)
    return U[:, :n_components], S[:n_components], V[:, :n_components]

# manual cross_entropy for single model output x (not necessarily distribution)
# and one-hot encoded label y, each a vector-Pytorch tensor

# needed for hvp (see below)

# X and Y are (N x x/y_dim) matrices, batch size N
# NOTE: changed to natural logarithm by Seyoon
def cross_entropy_manual(X, Y):
    # note X.shape[0] is the batch size
    X_softmax = X.exp() / X.exp().sum(axis=1).reshape((X.shape[0], 1))
    # TODO: check pytorch uses base 2
    return -(Y * torch.log(X_softmax)).sum()

# takes flattened data matrix X (shape (N by x_dim)) and one-hot targets
# matrix Y, clones them (num_batches) times vertically.

# US is the product of U and S
# X_mega: [N * num_components, x_dim]
# Y_mega: [N * num_components, y_dim=num_classes]
# US_mega: [xdim, N * num_components]
# V_mega: [xdim, N * num_components]
def make_megabatch(X, Y, US, V, num_components):
    # extract batch size
    batch_size = X.shape[0]

    X_mega = X.repeat(num_components, 1).detach().clone()
    Y_mega = Y.repeat(num_components, 1).detach().clone()
    US_mega = torch.repeat_interleave(US, repeats=batch_size, dim=1).detach().clone()
    V_mega = torch.repeat_interleave(V, repeats=batch_size, dim=1).detach().clone()

    torch.save(X_mega, 'X_mega.pt')
    torch.save(Y_mega, 'Y_mega.pt')
    torch.save(US_mega, 'US_mega.pt')
    torch.save(V_mega, 'V_mega.pt')

    return X_mega, Y_mega, US_mega, V_mega

def make_megabatch2(X, Y, US, V, num_components):
    # extract batch size
    batch_size = X.shape[0]
    total = num_components * batch_size

    X_multi = torch.zeros((num_components, batch_size, X.shape[1])).cuda()
    Y_multi = torch.zeros((num_components, batch_size, Y.shape[1])).cuda()
    US_multi = torch.zeros((batch_size, US.shape[0], num_components)).cuda()
    V_multi = torch.zeros((batch_size, V.shape[0], num_components)).cuda()

    X_multi[:] = X
    Y_multi[:] = Y
    US_multi[:] = US
    V_multi[:] = V

    X_multi = X_multi.reshape((total, X.shape[1])).detach().clone()
    Y_multi = Y_multi.reshape((total, Y.shape[1])).detach().clone()
    US_multi = US_multi.permute(2, 0, 1).reshape((total, US.shape[0])).t().detach().clone()
    V_multi = V_multi.permute(2, 0, 1).reshape((total, V.shape[0])).t().detach().clone()

    return X_multi, Y_multi, US_multi, V_multi

# given a pytorch function loss(x_i, y_i) (twice differentiable)
# and a neural network 'model', 
# compute matrix-vector products of the form:
# (\nabla_{x1 x2}^2 loss(model(x), y)) @ v

# data_shape is the original shape of the batch tensor (non-flattened
# iamges)

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

# computes quadratics of the form \sum w_i H(x_i, y_i) v_i over a batch, where H is a Hessian
# w.r.t. loss

# suppose the batch size is N

# X is a (N by x_dim) matrix, where x_dim is the dimensionality of the data
# Y is a (N by num_classes) matrix
# x1 is a string: 'x' to take the 1st derivative w.r.t. X, 'y' for 1st derivative
#      w.r.t. Y
# x2 is a string: ""           "" 2nd derivative w.r.t. X, 'y' for 2st derivative
#      w.r.t. Y
# V has the same shape as the variable corresponding to x1
# W has the same shape as the variable corresponding to x2

# see comments over hvp for further details
def hess_quadratic(loss, model, data_shape, X, Y, x1, x2, V, W):
    # setting up pytorch stuff to prepare for backprop
    Vvar = Variable(V, requires_grad=True)
    Wvar = Variable(W, requires_grad=True)

    # extract batch size
    N = X.shape[0]
    
    Xvar = Variable(X, requires_grad=True)
    Yvar = Variable(Y, requires_grad=True)
    model_eval = model(Xvar.reshape(data_shape))
    
    # choose which variable x1var corresponds to
    x1var = Xvar if x1=='x' else Yvar
    x2var = Xvar if x2=='x' else Yvar
    
    score = loss(model_eval, Yvar)

    # gradient w.r.t. entire batch 
    grad, = torch.autograd.grad(score, x1var, create_graph=True)
    # sum over batch elements (avg. at end)
    total = torch.sum(grad * Vvar)
    
    if Xvar.grad:
        Xvar.grad.data.zero_()
    if Yvar.grad:
        Yvar.grad.data.zero_()
    
    # NOTE: THIS WILL NOT ALLOW FURTHER BACKPROP, BRING create_graph=True BACK TO ALLOW THIS
    grad2, = torch.autograd.grad(total, x2var, create_graph=False, allow_unused=True)
    # sum over rows (different elements in batch)
    wHv = torch.sum(W * grad2)

    return (1/N)*wHv

# Computes a quadratic of the form w^T \sum_i H(x_i, y_i) v, where H is a Hessian w.r.t. loss
# v, w are vectors that come from an SVD, both have a leading dimension of 1 
#
# v has the same dimension as what x1 corresponds to (x_dim if 'x', num_classes if 'y')
# w has the same dimension as what x2 corresponds to
def hess_svd(loss, model, data_shape, X, Y, x1, x2, v, w):
    # setting up pytorch stuff to prepare for backprop
    vvar = Variable(v, requires_grad=True)
    wvar = Variable(w, requires_grad=True)

    # extract batch size
    N = X.shape[0]
    
    Xvar = Variable(X, requires_grad=True)
    Yvar = Variable(Y, requires_grad=True)
    model_eval = model(Xvar.reshape(data_shape))
    
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
    
    # NOTE: THIS WILL NOT ALLOW FURTHER BACKPROP, BRING create_graph=True BACK TO ALLOW THIS
    grad2, = torch.autograd.grad(total, x2var, create_graph=False, allow_unused=True)
    # sum over rows (different elements in batch)
    wHv = torch.sum(grad2.sum(axis=0) * wvar)

    # COMPARISON WITH OUTPUT FROM HESS_QUADRATIC
    #bigv = torch.zeros((X.shape[0], v.shape[1])).cuda()
    #bigv[:, :] = v
    #bigw = torch.zeros((X.shape[0], w.shape[1])).cuda()
    #bigw[:, :] = w

    #retval = hess_quadratic(loss, model, data_shape, X, Y, x1, x2, bigv, bigw)
    #print("COMPARISON INSIDE HESS_SVD", retval, 1/N * wHv)

    return (1/N)*wHv

# the equivalent Hessian quadratic form for the epsilon delta^2 terms:

# computes \delta^T (1/N) \sum_i \nabla_{x_ix_i}^2(log(S(model(x_i))_{class_val})) \delta
# See project overleaf for details.

# inputs identical to hess_svd, except for class_val
# class_val indicates which corresponding class the current matrix inner product
# is being taken w.r.t.
def hess_svd_ed2(class_val, model, data_shape, X, Y, x1, x2, v, w):
    # setting up pytorch stuff to prepare for backprop
    vvar = Variable(v, requires_grad=True)
    wvar = Variable(w, requires_grad=True)

    # extract batch size
    N = X.shape[0]
    
    Xvar = Variable(X, requires_grad=True)
    Yvar = Variable(Y, requires_grad=True)
    model_eval = model(Xvar.reshape(data_shape))

    # here is where we take only the class_val component
    model_eval_softmax = model_eval[:,class_val].exp().reshape((N,1)) / model_eval.exp().sum(axis=1).reshape((N, 1))

    # choose which variable x1var corresponds to
    x1var = Xvar if x1=='x' else Yvar
    x2var = Xvar if x2=='x' else Yvar
    
    f_val = torch.log(model_eval_softmax).sum()

    # gradient w.r.t. entire batch 
    grad, = torch.autograd.grad(f_val, x1var, create_graph=True)
    # sum over batch elements (avg. at end)
    total = torch.sum(grad.sum(axis=0) * vvar)
    
    if Xvar.grad:
        Xvar.grad.data.zero_()
    if Yvar.grad:
        Yvar.grad.data.zero_()
    
    # NOTE: THIS WILL NOT ALLOW FURTHER BACKPROP, BRING create_graph=True BACK TO ALLOW THIS
    grad2, = torch.autograd.grad(total, x2var, create_graph=False, allow_unused=True)
    # sum over rows (different elements in batch)
    wHv = torch.sum(grad2.sum(axis=0) * wvar)

    return (1/N)*wHv

# theta_bar is 3/4
# images and labels are not scaled down/flattened

# returns regularization (non-loss) terms
def taylor_loss(images, labels, model, mu_img, mu_y, Uxx, Sxx, Vxx, Uxy, Sxy, Vxy, T_U, T_S, T_V):
    # extract batch size
    N = images.shape[0]
    # extract number of total pixels for images
    img_size = int(images.numel() / N)
    # extract original batch shape
    batch_shape = images.shape

    # flatten input images
    images_flat = images.reshape((N, img_size))
    mu_img_flat = mu_img.reshape((1, img_size))

    num_classes = 10 # labels.max() + 1
    # Y is a stack of rows, where each row is the one_hot version
    # of the correct label
    Y = torch.zeros((N, num_classes)).cuda()
    Y[np.arange(N), labels] = 1

    # COMPUTE raw tilde loss (term 1)
    
    # we assume uniform distribution
    theta_bar = 0.75*torch.ones((1)).cuda()
    # matrix form for x_theta over whole batch
    Xt = (1 - theta_bar)*mu_img_flat + theta_bar*images_flat
    # same for y_tilde
    Yt = (1 - theta_bar)*mu_y + theta_bar*Y


    # torch.save(Xt, 'Xt.pt')
    # torch.save(Yt, 'Yt.pt')

    loss = (1/N)*cross_entropy_manual(model(Xt.reshape(batch_shape)), Yt)

    # COMPUTE delta delta term (term 2)

    # first compute the data-dependent part.
    V = (images_flat - mu_img_flat).detach().clone()
    # compute the data dependent component of inner product
    data_dependent = hess_quadratic(
        lambda x, y : cross_entropy_manual(x, y), model, batch_shape, Xt, Yt, 'x', 'x', V, V)
    
    # extract number of singular values extracted from global covariance matrix
    num_components = Sxx.numel()
    t1 = time.time()
    X_mega, Y_mega, US_mega, V_mega = make_megabatch2(Xt, Yt, Uxx * Sxx.reshape((1, num_components)), Vxx, num_components)
    #print((int(X_mega.numel()/img_size), batch_shape[1], batch_shape[2], batch_shape[3]))
    data_independent2 = num_components * hess_quadratic( # num_components is to fix normalisation
        lambda x, y : cross_entropy_manual(x, y), model, (int(X_mega.numel()/img_size), batch_shape[1], batch_shape[2], batch_shape[3]),
        X_mega, Y_mega, 'x', 'x', US_mega.t(), V_mega.t()
    )

    X_mega, Y_mega, US_mega, V_mega = make_megabatch(Xt, Yt, Uxx * Sxx.reshape((1, num_components)), Vxx, num_components)
    #print((int(X_mega.numel()/img_size), batch_shape[1], batch_shape[2], batch_shape[3]))
    data_independent3 = num_components * hess_quadratic( # num_components is to fix normalisation
        lambda x, y : cross_entropy_manual(x, y), model, (int(X_mega.numel()/img_size), batch_shape[1], batch_shape[2], batch_shape[3]),
        X_mega, Y_mega, 'x', 'x', US_mega.t(), V_mega.t()
    )
    t2 = time.time()

    data_independent = torch.zeros((1)).cuda()
    for i in range(num_components):
        data_independent += hess_svd(
            lambda x, y : cross_entropy_manual(x, y), model, batch_shape, Xt, Yt, 'x', 'x', Sxx[i]*Uxx[:,i].reshape((1, img_size)), Vxx[:,i].reshape((1, img_size)))

    X_mega = Xt.repeat(num_components, 1).detach().clone()
    Y_mega = Yt.repeat(num_components, 1).detach().clone()
    US_mega = torch.zeros((N * num_components, img_size)).cuda()
    V_mega = torch.zeros((N * num_components, img_size)).cuda()
    for i in range(num_components):
        US_mega[i*num_components:i*num_components+num_components, :] = Sxx[i]*Uxx[:,i].reshape((1, img_size))
        V_mega[i*num_components:i*num_components+num_components, :] = Vxx[:,i].reshape((1, img_size))

    data_independent4 = num_components * hess_quadratic( # num_components is to fix normalisation
        lambda x, y : cross_entropy_manual(x, y), model, (int(X_mega.numel()/img_size), batch_shape[1], batch_shape[2], batch_shape[3]),
        X_mega, Y_mega, 'x', 'x', US_mega, V_mega
    )
    

    t3 = time.time()
    print("COMPARISON", data_independent2, data_independent, data_independent4)
    var_half_mixup = 0.5**2 / 12
    gamma_squared = var_half_mixup + (1 - theta_bar)**2
    ddterm = 0.5*(var_half_mixup*data_dependent + gamma_squared * data_independent)

    # COMPUTE epsilon delta "cross-term" (term 3)

    # first compute the data-dependent part.
    W = (Y - mu_y).detach().clone()
    # compute the data dependent component of inner product
    data_dependent_cross = hess_quadratic(
        lambda x, y : cross_entropy_manual(x, y), model, batch_shape, Xt, Yt, 'x', 'y', V, W)
    
    # extract number of singular values extracted from global covariance matrix
    num_components = Sxy.numel()
    a, b, c, d = make_megabatch(Xt, Yt, Uxy * Sxy.reshape((1, num_components)), Vxy, num_components)

    data_independent_cross = torch.zeros((1)).cuda()
    for i in range(num_components):
        data_independent_cross += hess_svd(
            lambda x, y : cross_entropy_manual(x, y), model, batch_shape, Xt, Yt, 'x', 'y', Sxy[i]*Uxy[:,i].reshape((1, img_size)), Vxy[:,i].reshape((1, num_classes)))

    edterm = var_half_mixup*data_dependent_cross + gamma_squared * data_independent_cross

    # update num components
    num_components = T_S[0,:].numel()
    # COMPUTE epsilon delta delta "3-term" (term 4, new)
    hess_quad_innerprod = torch.zeros((1)).cuda()
    # sum over classes
    for i in range(num_classes):
        # sum over all compoments we take from T_a matrices
        for j in range(num_components):
            hess_quad_innerprod += hess_svd_ed2(
                i, model, batch_shape, Xt, Xt, 'x', 'x', T_S[i, j]*T_U[i, :, j].reshape((1, img_size)), T_V[i,:,j].reshape((1, img_size)))

    eddterm = -0.5 * ((1-theta_bar)**3) * hess_quad_innerprod
    return loss, ddterm, edterm, eddterm
