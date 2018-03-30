import torch


def flatten(x):
  """Flattens <x>, except for the batch dimension."""
  batch_size = x.size()[0]
  return x.view(batch_size, -1)


def randn_init(scale):
  """Returns a function that randomly initializes the module."""
  def func(module):
    for p in module.parameters():
      p = scale * torch.randn(*p.size())
  return func


#### METRICS ###########################################################
"""A metric is a function that takes two tensor arguments <outs> and <tgts>:
the values returned by the model, and the target values; and returns
a single value (like accuracy, mean squared error, ...)."""

def mean_squared_error(outs, tgts):
  return torch.mean(flatten((tgts - outs)**2), dim = 1)
