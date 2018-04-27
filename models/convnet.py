import torch
from torch import nn

from lib.functional import flatten
from .general import MyModule, Functional

from collections import OrderedDict


#### DESCRIPTION OF CONVNET ############################################

step_names = [
  "start",
  "conv11", "conv12", "conv13",
  "conv21", "conv22", "conv23",
  "conv31", "conv32", "conv33",
  "flatten", "dense1", "dense2"
]
prev_names = {
  step_names[i+1]: step_names[i] for i in range(len(step_names) - 1)
}
next_names = {
  step_names[i]: step_names[i+1] for i in range(len(step_names) - 1)
}
step_table = {
  **{"conv{}{}".format(i, j): nn.Conv2d for j in range(1, 4) for i in range(1, 4)},
  "flatten": Functional,
  "dense1": nn.Linear,
  "dense2": nn.Linear
}
step_args = {
  "conv11": [3, 6, 3],
  "conv12": [6, 12, 3],
  "conv13": [12, 12, 2],
  "conv21": [12, 24, 3],
  "conv22": [24, 48, 3],
  "conv23": [48, 48, 2],
  "conv31": [48, 96, 3],
  "conv32": [96, 192, 3],
  "conv33": [192, 192, 2],
  "flatten": [flatten],
  "dense1": [3072, 200],
  "dense2": [200, 10]
}
step_kwargs = {
  "conv11": {"padding": 1},
  "conv12": {"padding": 1},
  "conv13": {"stride": 2},
  "conv21": {"padding": 1},
  "conv22": {"padding": 1},
  "conv23": {"stride": 2},
  "conv31": {"padding": 1},
  "conv32": {"padding": 1},
  "conv33": {"stride": 2}
}
step_sizes = {
  "start": (3, 32, 32),
  "conv11": (6, 32, 32),
  "conv12": (12, 32, 32),
  "conv13": (12, 16, 16),
  "conv21": (24, 16, 16),
  "conv22": (48, 16, 16),
  "conv23": (48, 8, 8),
  "conv31": (96, 8, 8),
  "conv32": (192, 8, 8),
  "conv33": (192, 4, 4),
  "flatten": (3072,),
  "dense1": (200,),
  "dense2": (10,)
}


#### CREATION OF CONVNET ###############################################

def step(name):
  """Returns the module that should be added to our convnet in step
  named 'name'."""
  global step_table, step_args, step_kwargs
  if name not in step_table:
    return None
  args = step_args.get(name, [])
  kwargs = step_kwargs.get(name, {})
  return step_table[name](*args, **kwargs)


def layer_normed(name):
  """Returns the same module as in 'step', except that it is layer
  normalized."""
  module = step(name)
  if module and name not in ["flatten"]:
    module = nn.Sequential(module, nn.LayerNorm(step_sizes[name]))
  return module


def batch_normed(name):
  """Returns the same module as in 'step', except that it is batch
  normalized."""
  module = step(name)
  if module and name not in ["flatten"]:
    c = step_sizes[name][0]
    if name[:4] == "conv":
      norm_class = nn.BatchNorm2d
    elif name[:5] == "dense":
      norm_class = nn.BatchNorm1d
    module = nn.Sequential(module, norm_class(c))
  return module


def convnet(after_func, step_func = step):
  """Returns a convnet constructed from the given step and after-step
  functions."""
  pipeline = []
  global step_names
  for name in step_names:
    mod1 = step_func(name)
    if mod1:
      pipeline.append((name, mod1))
    
    # Do not use activation in the last layer.
    if name in next_names:
      name2 = "after_{}".format(name)
      mod2 = after_func(name)
      if mod2:
        pipeline.append((name2, mod2))
  
  return nn.Sequential(OrderedDict(pipeline))


def init_weights(gain, weight_norm = False):
  """Returns a function, that initializes our convnet's parameters
  recursively, using a variant of the xavier initialization with the
  given gain <gain>."""
  def func(module):
    if type(module) in [nn.Linear, nn.Conv2d]:
      nn.init.xavier_normal_(module.weight, gain)
      if weight_norm:
        nn.utils.weight_norm(module)
      if module.bias is not None:
        with torch.no_grad():
          module.bias.fill_(0)
  return func
