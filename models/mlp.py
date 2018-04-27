import torch
from torch import nn

from lib.functional import flatten
from .general import MyModule, Functional


"""Description of our MLP's architecture."""
num_layers = 20
layer_sizes = [3072] + [3000]*3 + [2000]*6 + [1000]*9 + [100, 10]


#### CREATION OF OUR MLP ###############################################

def step(in_size, out_size):
  """Simple dense (fully connected) layer, input size is <in_size>,
  output size is <out_size>."""
  return nn.Linear(in_size, out_size)


def weight_normed(in_size, out_size):
  """Returns the same module as in 'step', except that it is weight
  normalized."""
  module = step(in_size, out_size)
  if module:
    module = nn.weight_norm(module)
  return module


def layer_normed(in_size, out_size):
  """Returns the same module as in 'step', except that it is layer
  normalized."""
  module = step(in_size, out_size)
  if module:
    module = nn.Sequential(module, nn.LayerNorm(out_size))
  return module


def batch_normed(in_size, out_size):
  """Returns the same module as in 'step', except that it is batch
  normalized."""
  module = step(in_size, out_size)
  if module:
    module = nn.Sequential(module, nn.BatchNorm1d(out_size))
  return module


def mlp(after_func, step_func = step):
  """Returns a multilayer perceptron constructed from the given step
  and after-step functions."""
  pipeline = [Functional(flatten)]
  global num_layers, layer_sizes
  for i, in_size, out_size in zip(range(1, num_layers), layer_sizes[:-1], layer_sizes[1:]):
    mod1 = step_func(in_size, out_size)
    if mod1:
      pipeline.append(mod1)
    
    # Do not activate after last layer.
    if i < num_layers - 1:
      mod2 = after_func(in_size, out_size)
      if mod2:
        pipeline.append(mod2)
  
  return nn.Sequential(*pipeline)
