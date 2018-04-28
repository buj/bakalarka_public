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

def scaler_grad(module, grad_ins, grad_outs):
  """Adjusts the gradient with respect to <module>.ax and <module>.ay,
  where <module> should be a Scaler module."""
  with torch.no_grad():
    module.ax.grad *= (module.eps**2 + module.ax.data**2)**0.5
    module.ay.grad *= (module.eps**2 + module.ay.data**2)**0.5

class Scaler(MyModule):
  """Can wrap an activation function. Enables scaling of both the input
  and the output."""
  
  def __init__(self, sub, eps = 10**-6):
    """<sub> is the underlying activation that is to be scaled. <eps>
    is None if we don't want to adjust the gradient, and any real value
    otherwise (used in scaler grad). The larger <eps> is, the easier it
    is to change the sign of 'self.ax' and 'self.ay'."""
    self.sub = sub
    self.ax = nn.Parameter(torch.ones(1))
    self.ay = nn.Parameter(torch.ones(1))
    if eps is not None:
      self.eps = eps
      self.register_backward_hook(scaler_grad)
  
  def forward(self, x):
    return self.ay * self.sub(x * self.ax)


def zoomer_grad(module, grad_ins, grad_outs):
  """Adjusts the gradient with respect to <module>.k. <module>
  should be a Zoomer module."""
  with torch.no_grad():
    module.k.grad /= module.k.data

class Zoomer(MyModule):
  """Can wrap an activation function. Enables scaling of both the input
  and the output. The scalings are inverse, and the result is a 'zoom-in'
  or 'zoom-out' of the activation function."""
  
  def __init__(self, sub, adjust_grad = True):
    self.sub = sub
    self.k = nn.Parameter(torch.zeros(1))
    if adjust_grad:
      self.register_backward_hook(zoomer_grad)
  
  def forward(self, x):
    return torch.exp(-self.k) * self.sub(x * torch.exp(self.k))


class Centerer(MyModule):
  """Can wrap an activation function. Enables centering of both the input
  and the output."""
  
  def __init__(self, sub):
    self.sub = sub
    self.bx = nn.Parameter(torch.zeros(1))
    self.by = nn.Parameter(torch.zeros(1))
  
  def forward(self, x):
    return self.sub(x + self.bx) + self.by


class Tracker(MyModule):
  """Can wrap an activation function. Enables centering of both the input
  and the output. The adjustments are inverse with respect to each other."""
  
  def __init__(self, sub):
    self.sub = sub
    self.b = nn.Parameter(torch.zeros(1))
  
  def forward(self, x):
    return self.sub(x + self.b) - self.sub(self.b)


#### SPECIFIC PARAMETERIZED FUNCTIONS ##################################

from lib.functional import parameterized_sgnlog


class ParameterizedSgnlog(MyModule):
  """Implements the parameterized sgnlog."""
  
  def __init__(self):
    # Start as a linear function.
    self.p = nn.Parameter(torch.ones(1))
  
  def forward(self, x):
    return parameterized_sgnlog(x, p)
