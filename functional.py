import torch
from torch.autograd import Function

import math


#### ACTIVATIONS #######################################################

def sgnlog(x):
  """Elementwise sgnlog."""
  return torch.sign(x) * torch.log(1 + torch.abs(x))


def parameterized_sgnlog(x, p):
  """Elementwise parameterized sgnlog, where <p> is the parameter."""
  y = x.new_empty(x.shape, requires_grad = True)
  
  # Case 1: p == 0
  case1 = (p == 0)
  x1 = x[case1]
  y[case1] = sgnlog(x1)
  
  # Case2: p != 0 and p <= 1  
  case2 = (p <= 1) * (p != 0)
  x2 = x[case2]
  p2 = p[case2]
  y[case2] = torch.sign(x2) * ((1 + torch.abs(x2))**p2 - 1) / p2
  
  # Case 3: p > 1  
  case3 = (p > 1)
  x3 = x[case3]
  p3 = p[case3]
  y[case3] = torch.sign(x3) * ((1 + torch.abs(x3)/p3) ** p3 - 1)
  
  return y


#### UTILS #############################################################

def flatten(x):
  """Flattens <x>, except for the batch dimension."""
  batch_size = x.size()[0]
  return x.view(batch_size, -1)


#### METRICS ###########################################################
"""A metric is a function that takes two tensor arguments <outs> and <tgts>:
the values returned by the model, and the target values; and returns
a single value (like accuracy, mean squared error, ...)."""

def mean_squared_error(outs, tgts):
  return torch.mean(flatten((tgts - outs)**2), dim = 1)


def cross_entropy(outs, tgts):
  """<tgts> is a number specifying which category is the right one.
  It is NOT a one-hot encoded target vector."""
  temp = torch.sum(flatten(torch.exp(outs)), dim = 1)
  i = outs.new_empty(0, dtype = torch.long)
  torch.arange(outs.shape[0], out = i)
  return -outs[i, tgts] + torch.log(temp)


def accuracy(outs, tgts):
  """<tgts> is a number specifying which category is the right one.
  <outs> is a vector of values that represent our estimates of 'how
  likely is it this category'; the higher value, the more likely."""
  i = outs.new_empty(0, dtype = torch.long)
  torch.arange(outs.shape[0], out = i)
  outs = flatten(outs)
  res = outs.new_empty(0, dtype = torch.float)
  torch.prod(outs <= outs[i, tgts].view(-1, 1), dim = 1, out = res)
  return res
