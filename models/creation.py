import torch
from torch import nn

from .general import MyModule, Functional

from collections import OrderedDict

import logging


#### NORM FUNCS ########################################################

norm_mapper = {}


def is_norm_func(f):
  """Registers <f> in the norm_mapper."""
  name = f.__name__
  global norm_mapper
  if name in norm_mapper:
    logging.info("Already have a function named %s in norm_mapper! Aborting", name)
  else:
    norm_mapper[name] = f
  return f


@is_norm_func
def layer_norm(arch, name):
  """Returns a layer normalization module that can be applied right
  after step <name>."""
  if name not in ["flatten"]:
    return nn.LayerNorm(arch.sizes[name])


@is_norm_func
def batch_norm(arch, name):
  """Returns a batch normalization module that can be applied right
  after step <name>."""
  if name not in ["flatten"]:
    c = arch.sizes[name][0]
    if name[:4] == "conv":
      return nn.BatchNorm2d(c)
    elif name[:5] == "dense":
      return nn.BatchNorm1d(c)


#### AFTER FUNCS #######################################################

from .general import ParameterizedSgnlog, Scaler, Zoomer, Centerer, Tracker, NegPoser
from lib.functional import sgnlog as sgnlog_func, linsin2 as linsin2_func


after_mapper = {}


def is_after_func(f):
  """Registers <f> in the after_mapper."""
  name = f.__name__
  global after_mapper
  if name in after_mapper:
    logging.info("Already have a function named %s in after_mapper! Aborting", name)
  else:
    after_mapper[name] = f
  return f


#### RELU activations

@is_after_func
def relu(*args):
  return Functional(nn.functional.relu)

@is_after_func
def sc_relu(*args):
  return Scaler(Centerer(relu()))

@is_after_func
def zt_relu(*args):
  return Zoomer(Tracker(relu()))


#### SGNLOG activations

@is_after_func
def sgnlog(*args):
  return Functional(sgnlog_func)

@is_after_func
def sc_sgnlog(*args):
  return Scaler(Centerer(sgnlog()))

@is_after_func
def zt_sgnlog(*args):
  return Zoomer(Tracker(sgnlog()))

@is_after_func
def negpos_sgnlog(*args):
  return NegPoser(Zoomer(sgnlog()), Zoomer(sgnlog()))

@is_after_func
def parameterized_sgnlog(*args):
  return ParameterizedSgnlog()


#### TANH activations

@is_after_func
def tanh(*args):
  return Functional(nn.functional.tanh)

@is_after_func
def sc_tanh(*args):
  return Scaler(Centerer(tanh()))

@is_after_func
def zt_tanh(*args):
  return Zoomer(Tracker(tanh()))


#### LINSIN activations

@is_after_func
def linsin2(*args):
  return Functional(linsin2_func)

@is_after_func
def sc_linsin2(*args):
  return Scaler(Centerer(linsin2()))

@is_after_func
def zt_linsin2(*args):
  return Zoomer(Tracker(linsin2()))


#### CREATION OF NET ###################################################

net_mapper = {}


def step(arch, name):
  """Returns the module that should be added to our net in step
  named 'name'."""
  if name not in arch.table:
    return None
  args = arch.args.get(name, [])
  kwargs = arch.kwargs.get(name, {})
  return arch.table[name](*args, **kwargs)


class NetDescription:
  """Contains the description of the convolutional network: parameters
  and steps that construct the net."""
  
  def __init__(self, arch_name, names, table, args, kwargs, sizes):
    """
    <arch_name> is the name of the architecture (the description).
    <names> contains step names, in the order in which they will be
      executed.
    <prev_names> contains for each step the name of the previous step.
    <next_names> contains the next step's name.
    <table> contains the module class for each step.
    <args> contains the arguments passed to the class constructor.
    <kwargs> contains the keyword arguments passed to the class constructor.
    <sizes> contains the state's shape after each operation.
    """
    self.arch_name = arch_name
    global net_mapper
    if arch_name in net_mapper:
      logging.info("Already have a function named %s in net_mapper! Aborting", arch_name)
    else:
      net_mapper[arch_name] = self
    
    self.names = names
    self.prev_names = {names[i+1]: names[i] for i in range(len(names) - 1)}
    self.next_names = {names[i]: names[i+1] for i in range(len(names) - 1)}
    self.table = table
    self.args = args
    self.kwargs = kwargs
    self.sizes = sizes
  
  def __call__(self, after_func = None, norm_func = None):
    """Returns a constructed net."""
    pipeline = []
    for name in self.names:
      mod0 = step(self, name)
      if mod0:
        pipeline.append((name, mod0))
      
      if norm_func:
        # Do not use norm_func after the last step.
        if name in self.next_names:
          name1 = "norm_{}".format(name)
          mod1 = norm_func(self, name)
          if mod1:
            pipeline.append((name1, mod1))
      
      if after_func:
        # Do not use activation in the last layer.
        if name in self.next_names:
          name2 = "after_{}".format(name)
          mod2 = after_func(self, name)
          if mod2:
            pipeline.append((name2, mod2))
    
    return nn.Sequential(OrderedDict(pipeline))


def init_weights(gain, weight_norm = False):
  """Returns a function, that initializes our net's parameters
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
