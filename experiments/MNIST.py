import torch

from . import *

import logging


#### BOILERPLATE #######################################################

from lib.testdata import original_MNIST
from lib.functional import cross_entropy, accuracy
from lib.util import prefix, prefix_as


ctx.train_data, ctx.val_data = original_MNIST()
ctx.criterion = torch.nn.CrossEntropyLoss()
ctx.metrics = [cross_entropy, accuracy]


prefix[:] = ["experiments", "MNIST"]
suffix[:] = t_cent


#### ARCHITECTURES #####################################################

from torch import nn
from lib.functional import flatten


def from_list(l):
  """If <l> is a list, return a Sequential containing its modules."""
  if type(l) is list:
    return nn.Sequential(*l)
  return l


class RecurrentMNIST(nn.Module):
  """A classifier that processes the image pixel by pixel, in a specified
  order."""
  
  def __init__(self, order, **kwargs):
    """Construct a variant of the net, based on the constructors passed
    in <kwargs>. The order in which we process the image is specified
    by <order>."""
    super().__init__()
    self.order = order
    
    # Dense layer.
    self.input_dense = nn.Linear(1, 100, bias = False)
    self.state_dense = nn.Linear(100, 100)
    with torch.no_grad():
      self.input_dense.weight.data = 0.001 * torch.randn(100, 1)
      self.state_dense.weight.data = torch.eye(100)
      self.input_dense.bias.fill_(0.0)
      self.state_dense.bias.fill_(0.0)
    
    # Other, bonus layers.
    self.before = from_list(kwargs["before"]((100,)))
    self.act = from_list(kwargs["act"]((100,)))
    self.after = from_list(kwargs["after"]((100,)))
    
    # Output dense layer.
    self.output = nn.Linear(100, 10)
    nn.init.xavier_normal_(self.output.weight)
    with torch.no_grad():
      self.output.bias.fill_(0.0)
  
  def forward(self, x):
    """Flattens <x>, permutes it, and processes it pixel by pixel."""
    x = flatten(x)[:, self.order]
    curr = x.new_zeros(x.shape[0], 100)
    for i in range(784):
      ins = x[:, i].view(-1, 1)
      curr = self.state_dense(curr) + self.input_dense(ins)
      curr = self.after(self.act(self.before(curr)))
    return self.output(curr)


def recurrent(order = torch.arange(784, dtype = torch.int64)):
  """Returns a net constructor that constructs a RecurrentMNIST with
  the specified order of pixels."""
  def res(**kwargs):
    return RecurrentMNIST(order, **kwargs)
  res.__name__ = "recurrent"
  return res


#### EXPERIMENTS #######################################################

from lib.models.constructors import *


#### RECURRENT NETWORK #################################################

seeds = [6356404336168542494, 8299105321489685025, 1024307847847789434, 6978523842280092309]
