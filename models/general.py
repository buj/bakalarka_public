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

class Scaler(MyModule):
  """Can wrap an activation function. Enables scaling of both the input
  and the output."""
  
  def __init__(self, sub):
    self.sub = sub
    self.ax = nn.Parameter(torch.ones(1))
    self.ay = nn.Parameter(torch.ones(1))
  
  def forward(self, x):
    return self.ay * self.sub(x * self.ax)


class Zoomer(MyModule):
  """Can wrap an activation function. Enables scaling of both the input
  and the output. The scalings are inverse, and the result is a 'zoom-in'
  or 'zoom-out' of the activation function."""
  
  def __init__(self, sub):
    self.sub = sub
    self.k = nn.Parameter(torch.zeros(1))
  
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


class Shaker(MyModule):
  """Can wrap an activation function. Enables centering of both the input
  and the output. The adjustments are inverse with respect to each other."""
  
  def __init__(self, sub):
    self.sub = sub
    self.b = nn.Parameter(torch.zeros(1))
  
  def forward(self, x):
    return self.sub(x + self.b) - self.sub(self.b)
