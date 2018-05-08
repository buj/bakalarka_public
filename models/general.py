import torch
from torch import nn
from torch.autograd import Variable


class Functional(nn.Module):
  """A module that implements an arbitrary unary function. It has
  no trainable parameters."""
  
  def __init__(self, func):
    super().__init__()
    self.func = func
  
  def forward(self, x):
    return self.func(x)


#### ELEMENTWISE NORMALIZATIONS ########################################

def scaler_grad(param, eps = 10**-6):
  """Returns a hook that adjusts the gradient with respect to <param>.
  Used by the ActScaler and Scaler modules."""
  def hook(grad):
    with torch.no_grad():
      return grad * (eps**2 + param**2)**0.5
  return hook


class Scaler(nn.Module):
  """Scales the input, each element has its owen scaling factor (as
  opposed to a global scaling factor)."""
  
  def __init__(self, in_size, eps = 10**-6):
    """<in_size> is the size of the input."""
    super().__init__()
    self.weight = nn.Parameter(torch.ones(*in_size))
    if eps is not None:
      self.eps = eps
      self.weight.register_hook(scaler_grad(self.weight, self.eps))
  
  def forward(self, x):
    return x * self.weight


def pos_scaler_grad(param):
  """Returns a hook that adjusts the gradient with respect to <param>.
  Used by the PositiveScaler module."""
  def hook(grad):
    with torch.no_grad():
      return grad * torch.exp(-param)
  return hook


class PositiveScaler(nn.Module):
  """Scales the input, each element has its own positive scaling factor
  (as opposed to a global scaling factor)."""
  
  def __init__(self, in_size, eps = 10**-6):
    """<in_size> is the size of the input."""
    super().__init__()
    self.weight = nn.Parameter(torch.zeros(*in_size))
    if eps is not None:
      self.eps = eps
      self.weight.register_hook(pos_scaler_grad(self.weight))
  
  def forward(self, x):
    return x * torch.exp(self.weight)


class Shifter(nn.Module):
  """Shifts the input, each element has its own bias (as opposed to
  a global bias)."""
  
  def __init__(self, in_size):
    """<in_size> is the size of the input."""
    super().__init__()
    self.weight = nn.Parameter(torch.zeros(*in_size))
  
  def forward(self, x):
    return x + self.weight


class NegPoser(nn.Module):
  """Wraps two modules. The first determines the behaviour
  on negative values, the second determines the behaviour on
  positive values. For zero, returns the average of the two."""
  
  def __init__(self, sub1, sub2):
    super().__init__()
    self.sub1 = sub1
    self.sub2 = sub2
  
  def forward(self, x):
    neg = self.sub1(x)
    pos = self.sub2(x)
    sgn = torch.sign(x)
    return (1+sgn)/2 * pos + (1-sgn)/2 * neg


#### ACTIVATION FUNCTION WARPERS #######################################

class ActScaler(nn.Module):
  """Can wrap an activation function. Enables scaling of both the input
  and the output.
  
  Is applied layerwise, that is, it has one parameter for an entire
  layer. It can thus be seen as a form of layer normalization."""
  
  def __init__(self, sub, eps = 10**-6, ins = True, outs = True):
    """<sub> is the underlying activation that is to be scaled. <eps>
    is None if we don't want to adjust the gradient, and any real value
    otherwise (used in scaler grad). The larger <eps> is, the easier it
    is to change the sign of 'self.ax' and 'self.ay'."""
    super().__init__()
    self.sub = sub
    self.ax = nn.Parameter(torch.ones(1)) if ins else 1
    self.ay = nn.Parameter(torch.ones(1)) if outs else 1
    if eps is not None:
      self.eps = eps
      if ins:
        self.ax.register_hook(scaler_grad(self.ax, self.eps))
      if outs:
        self.ay.register_hook(scaler_grad(self.ay, self.eps))
  
  def forward(self, x):
    return self.ay * self.sub(x * self.ax)


class ActZoomer(nn.Module):
  """Can wrap an activation function. Enables scaling of both the input
  and the output. The scalings are inverse, and the result is a 'zoom-in'
  or 'zoom-out' of the activation function.
  
  By default, is applied layerwise, that is, it has one parameter for
  an entire layer. It can thus be seen as a form of layer normalization."""
  
  def __init__(self, sub, in_size = (1,)):
    """If <in_size> is not provided, one parameter is used for the
    entire layer."""
    super().__init__()
    self.sub = sub
    self.k = nn.Parameter(torch.zeros(*in_size))
  
  def forward(self, x):
    return torch.exp(-self.k) * self.sub(x * torch.exp(self.k))


class ActCenterer(nn.Module):
  """Can wrap an activation function. Enables centering of both the input
  and the output.
  
  Is applied layerwise, that is, it has one parameter for an entire
  layer. It can thus be seen as a form of layer normalization."""
  
  def __init__(self, sub, ins = True, outs = True):
    super().__init__()
    self.sub = sub
    self.bx = nn.Parameter(torch.zeros(1)) if ins else 0
    self.by = nn.Parameter(torch.zeros(1)) if outs else 0
  
  def forward(self, x):
    return self.sub(x + self.bx) + self.by


class ActTracker(nn.Module):
  """Can wrap an activation function. Enables centering of both the input
  and the output. The adjustments are inverse with respect to each other.
  
  By default, is applied layerwise, that is, it has one parameter for
  an entire layer. It can thus be seen as a form of layer normalization."""
  
  def __init__(self, sub, in_size = (1,)):
    """If <in_size> is not provided, one parameter is used for the
    entire layer."""
    super().__init__()
    self.sub = sub
    self.b = nn.Parameter(torch.zeros(*in_size))
  
  def forward(self, x):
    return self.sub(x + self.b) - self.sub(self.b)


#### SPECIFIC PARAMETERIZED ACTIVATIONS ################################

from lib.functional import parameterized_sgnlog


class ParameterizedSgnlog(nn.Module):
  """Implements the parameterized sgnlog."""
  
  def __init__(self, in_size):
    """<in_size> is the size of the input, the number of nodes. Each
    node has its own parameter."""
    super().__init__()
    self.p = nn.Parameter(torch.ones(in_size))
  
  def forward(self, x):
    return parameterized_sgnlog(x, self.p)


#### RANDOM STUFF ######################################################

class IoLinear(nn.Module):
  """Similar to a linear layer, except that each input has a weight
  that influences all edges coming from that input, and each output
  has a weight that influences edges coming into it."""
  
  def __init__(self, in_size, out_size):
    super().__init__()
    self.weight = nn.Parameter(torch.empty(out_size, in_size))
    
    # These are skipped by used init procedures, so we initialize it here.
    self.in_weight = nn.Parameter(torch.zeros(in_size))
    self.out_weight = nn.Parameter(torch.zeros(out_size))
    
    # Give the weights hooks, so that they learn evenly and that
    # when compared to bias, the ratio of learning is the same as
    # in linear layers.
    self.weight.register_hook(lambda grad: grad / 3)
    self.in_weight.register_hook(lambda grad: grad / out_size / 3)
    self.out_weight.register_hook(lambda grad: grad / in_size / 3)
    
    self.bias = nn.Parameter(torch.empty(out_size))
  
  def forward(self, x):
    W = torch.unsqueeze(self.in_weight, 0) + self.weight + torch.unsqueeze(self.out_weight, 1)
    return nn.functional.linear(x, W, self.bias)
