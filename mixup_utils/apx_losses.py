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
def hvp(g, x, y, v):
    xvar = Variable(x, requires_grad=True)
    yvar = Variable(y, requires_grad=True)
    vvar = Variable(v, requires_grad=True)
    
    score = g(xvar, yvar)
    
    grad, = torch.autograd.grad(score, yvar, create_graph=True)
    #print(grad)
    total = torch.sum(grad * vvar)
    #print(total)
    
    if xvar.grad:
        xvar.grad.data.zero_()
    if yvar.grad:
        yvar.grad.data.zero_()
        
    grad2, = torch.autograd.grad(total, xvar, create_graph=True, allow_unused=True)
    return grad2

# double sum
def doublesum(l, x, y):

    return -1