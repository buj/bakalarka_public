import torch


#### ACTIVATIONS #######################################################

def sgnlog(x):
  """Elementwise sgnlog."""
  return torch.sign(x) * torch.log(1 + torch.abs(x))


def parameterized_sgnlog(x, p):
  """Elementwise parameterized sgnlog, where <p> is the parameter."""
  if p == 0:
    return sgnlog(x)
  if p <= 1:
    return torch.sign(x) * ((1 + torch.abs(x))**p - 1) / p
  return torch.sign(x) * ((1 + torch.abs(x)/p) ** p - 1)


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
  i = torch.LongTensor()
  torch.arange(outs.shape[0], out = i)
  return -outs[i, tgts] + torch.log(temp)
