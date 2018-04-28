import torch
from torch import nn
from torch.autograd import Variable


class MyModule(nn.Module):
  """Custom nn_module with enhanced capabilities that are missing in
  pytorch."""
  
  def __call__(self, x):
    # Converts <x> to Variable if necessary, because !@#%ing pytorch
    # can't do stuff when Tensors and Variables are mixed.
    if not isinstance(x, Variable):
      old_train = self.training
      self.train(False)
      x = Variable(x)
      res = super().__call__(x).data
      self.train(old_train)
      return res
    return super().__call__(x)


class Functional(MyModule):
  """A module that implements an arbitrary unary function. It has
  no trainable parameters."""
  
  def __init__(self, func):
    super().__init__()
    self.func = func
  
  def forward(self, x):
    return self.func(x)


#### GENERAL PAREMETERIZED ACTIVATIONS #################################

def scaler_grad(param, eps):
  """Returns a hook that adjusts the gradient with respect to <param>.
  Used by the Scaler module."""
  def hook(grad):
    with torch.no_grad():
      return grad * (eps**2 + param**2)**0.5
  return hook

class Scaler(MyModule):
  """Can wrap an activation function. Enables scaling of both the input
  and the output."""
  
  def __init__(self, sub, eps = 10**-6):
    """<sub> is the underlying activation that is to be scaled. <eps>
    is None if we don't want to adjust the gradient, and any real value
    otherwise (used in scaler grad). The larger <eps> is, the easier it
    is to change the sign of 'self.ax' and 'self.ay'."""
    super().__init__()
    self.sub = sub
    self.ax = nn.Parameter(torch.ones(1))
    self.ay = nn.Parameter(torch.ones(1))
    if eps is not None:
      self.eps = eps
      self.ax.register_hook(scaler_grad(self.ax, self.eps))
      self.ay.register_hook(scaler_grad(self.ay, self.eps))
  
  def forward(self, x):
    return self.ay * self.sub(x * self.ax)


def zoomer_grad(param):
  """Returns a hook that adjusts the gradient with respect to <k>.
  Used by the Zoomer module."""
  def hook(grad):
    with torch.no_grad():
      return grad / param
  return hook

class Zoomer(MyModule):
  """Can wrap an activation function. Enables scaling of both the input
  and the output. The scalings are inverse, and the result is a 'zoom-in'
  or 'zoom-out' of the activation function."""
  
  def __init__(self, sub, adjust_grad = True):
    super().__init__()
    self.sub = sub
    self.k = nn.Parameter(torch.zeros(1))
    if adjust_grad:
      self.k.register_hook(zoomer_grad(self.k))
  
  def forward(self, x):
    return torch.exp(-self.k) * self.sub(x * torch.exp(self.k))


class Centerer(MyModule):
  """Can wrap an activation function. Enables centering of both the input
  and the output."""
  
  def __init__(self, sub):
    super().__init__()
    self.sub = sub
    self.bx = nn.Parameter(torch.zeros(1))
    self.by = nn.Parameter(torch.zeros(1))
  
  def forward(self, x):
    return self.sub(x + self.bx) + self.by


class Tracker(MyModule):
  """Can wrap an activation function. Enables centering of both the input
  and the output. The adjustments are inverse with respect to each other."""
  
  def __init__(self, sub):
    super().__init__()
    self.sub = sub
    self.b = nn.Parameter(torch.zeros(1))
  
  def forward(self, x):
    return self.sub(x + self.b) - self.sub(self.b)


#### SPECIFIC PARAMETERIZED FUNCTIONS ##################################

from lib.functional import parameterized_sgnlog


class ParameterizedSgnlog(MyModule):
  """Implements the parameterized sgnlog."""
  
  def __init__(self):
    super().__init__()
    # Start as a linear function.
    self.p = nn.Parameter(torch.ones(1))
  
  def forward(self, x):
    return parameterized_sgnlog(x, p)
