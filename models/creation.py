import torch
from torch import nn

from lib.functional import sgnlog as sgnlog_func
from .general import MyModule, Functional

from collections import OrderedDict

import logging


#### STEP FUNCS ########################################################

step_mapper = {}


def is_step_func(f):
  """Registers <f> in the step_mapper."""
  name = f.__name__
  global step_mapper
  if name in step_mapper:
    logging.info("Already have a function named %s in step_mapper! Aborting", name)
  else:
    step_mapper[name] = f
  return f


@is_step_func
def step(arch, name):
  """Returns the module that should be added to our net in step
  named 'name'."""
  if name not in arch.table:
    return None
  args = arch.args.get(name, [])
  kwargs = arch.kwargs.get(name, {})
  return arch.table[name](*args, **kwargs)


@is_step_func
def layer_normed(arch, name):
  """Returns the same module as in 'step', except that it is layer
  normalized."""
  module = step(arch, name)
  if module and name not in ["flatten"]:
    module = nn.Sequential(module, nn.LayerNorm(arch.sizes[name]))
  return module


@is_step_func
def batch_normed(arch, name):
  """Returns the same module as in 'step', except that it is batch
  normalized."""
  module = step(arch, name)
  if module and name not in ["flatten"]:
    c = arch.sizes[name][0]
    if name[:4] == "conv":
      norm_class = nn.BatchNorm2d
    elif name[:5] == "dense":
      norm_class = nn.BatchNorm1d
    module = nn.Sequential(module, norm_class(c))
  return module


#### AFTER FUNCS #######################################################

from .general import ParameterizedSgnlog, Scaler, Zoomer, Centerer, Tracker


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
def relu(arch, name):
  return Functional(nn.functional.relu)

@is_after_func
def sc_relu(arch, name):
  return Scaler(Centerer(Functional(nn.functional.relu)))

@is_after_func
def zt_relu(arch, name):
  return Zoomer(Tracker(Functional(nn.functional.relu)))


#### SGNLOG activations

@is_after_func
def sgnlog(arch, name):
  return Functional(sgnlog_func)

@is_after_func
def sc_sgnlog(arch, name):
  return Scaler(Centerer(Functional(sgnlog_func)))

@is_after_func
def zt_sgnlog(arch, name):
  return Zoomer(Tracker(Functional(sgnlog_func)))

@is_after_func
def parameterized_sgnlog(arch, name):
  return ParameterizedSgnlog()


#### TANH activations

@is_after_func
def tanh(arch, name):
  return Functional(nn.functional.tanh)

@is_after_func
def sc_tanh(arch, name):
  return Scaler(Centerer(Functional(nn.functional.tanh)))

@is_after_func
def zt_tanh(arch, name):
  return Zoomer(Tracker(Functional(nn.functional.tanh)))


#### CREATION OF NET ###################################################

net_mapper = {}


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
    
    # Default values for 'after_func' and 'step_func' during construction.
    global step
    self.after_func = None
    self.step_func = step
  
  def __call__(self, after_func = None, step_func = None):
    """Returns a constructed net."""
    if not after_func:
      after_func = self.after_func
    if not step_func:
      step_func = self.step_func
    
    pipeline = []
    for name in self.names:
      mod1 = step_func(self, name)
      if mod1:
        pipeline.append((name, mod1))
      
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
