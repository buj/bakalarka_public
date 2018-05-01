import torch
from torch import nn

from .general import MyModule, Functional

import random
from collections import OrderedDict

import logging


#### NORM FUNCS ########################################################

from .general import ElementwiseScaler, PositiveElementwiseScaler, ElementwiseShifter, NegPoser


norm_mapper = {}

def str_to_norm(f):
  """If <f> is a string, return the corresponding entry from norm_mapper.
  Otherwise, assume <f> is a normalization_function itself."""
  if type(f) == str:
    f = norm_mapper.get(f, None)
  return f


def is_norm_func(f):
  """Registers <f> in the norm_mapper."""
  name = f.__name__
  global norm_mapper
  if name in norm_mapper:
    logging.info("Already have a function named %s in norm_mapper! Aborting", name)
  else:
    norm_mapper[name] = f
  return f


def non_flatten_norm_func(f):
  """Registers <f> in the norm mapper, and in addition to that,
  makes it ignore the 'flatten' step."""
  def func(*args):
    if len(args) < 2 or args[1] not in ["flatten"]:
      return f(*args)
  func.__name__ = f.__name__
  return is_norm_func(func)


#### LAYER NORMS

@non_flatten_norm_func
def layer_norm(arch, name):
  """Returns a layer normalization module that can be applied right
  after step <name>."""
  return nn.LayerNorm(arch.sizes[name])


#### ELEMENTWISE NORMS

@non_flatten_norm_func
def batch_norm(arch, name):
  """Returns a batch normalization module that can be applied right
  after step <name>."""
  c = arch.sizes[name][0]
  if name[:4] == "conv":
    return nn.BatchNorm2d(c)
  elif name[:5] == "dense":
    return nn.BatchNorm1d(c)



@non_flatten_norm_func
def elem_scale(arch, name):
  """Returns an elemntwise scaler module that can be applied right
  after step <name>."""
  c = arch.sizes[name]
  return ElementwiseScaler(c)

@non_flatten_norm_func
def elem_scale2(arch, name):
  """Returns an elemntwise scaler module that can be applied right
  after step <name>. Does NOT scale the gradient proportionally to the
  size of the parameter."""
  c = arch.sizes[name]
  return ElementwiseScaler(c, None)



@non_flatten_norm_func
def pos_elem_scale(arch, name):
  """Returns a positive elementwise scaler module that can be applied
  right after step <name>."""
  c = arch.sizes[name]
  return PositiveElementwiseScaler(c)


@non_flatten_norm_func
def imp_norm(arch, name):
  """Returns the implicit normalization module that can be applied
  right after step <name>."""
  c = arch.sizes[name]
  return nn.Sequential(ElementwiseShifter(c), ElementwiseScaler(c))

@non_flatten_norm_func
def imp_norm2(arch, name):
  """Returns the implicit normalization module that can be applied
  right after step <name>. Does NOT scale the gradient proportionally
  to the size of the parameter."""
  c = arch.sizes[name]
  return nn.Sequential(ElementwiseShifter(c), ElementwiseScaler(c, None))



@non_flatten_norm_func
def np_elem_scale(arch, name):
  """Returns the negpos'd elementwise scaler module. It has separate
  scaling factors for positive and negative values. Can be applied
  right after ste <name>."""
  c = arch.sizes[name]
  return NegPoser(ElementwiseScaler(c), ElementwiseScaler(c))

@non_flatten_norm_func
def np_elem_scale2(arch, name):
  """Returns the negpos'd elementwise scaler module. It has separate
  scaling factors for positive and negative values. Can be applied
  right after ste <name>."""
  c = arch.sizes[name]
  return NegPoser(ElementwiseScaler(c, None), ElementwiseScaler(c, None))



@non_flatten_norm_func
def np_imp_norm(arch, name):
  """Returns the negpos'd implicit normalization module that can be
  applied right after step <name>. It has separate scaling factors for
  positive and negative values."""
  c = arch.sizes[name]
  np_scaler = NegPoser(ElementwiseScaler(c), ElementwiseScaler(c))
  return nn.Sequential(ElementwiseShifter(c), np_scaler)

@non_flatten_norm_func
def np_imp_norm2(arch, name):
  """Returns the negpos'd implicit normalization module that can be
  applied right after step <name>. It has separate scaling factors for
  positive and negative values. Does NOT scale the gradient
  proportionally to the size of the parameter."""
  c = arch.sizes[name]
  np_scaler = NegPoser(ElementwiseScaler(c, None), ElementwiseScaler(c, None))
  return nn.Sequential(ElementwiseShifter(c), np_scaler)


#### ACT FUNCS #######################################################

from lib.functional import sgnlog as sgnlog_func
from .general import ParameterizedSgnlog, Zoomer, Tracker, NegPoser


act_mapper = {}

def str_to_act(f):
  """If <f> is a string, return the corresponding entry from act_mapper.
  Otherwise, assume <f> is an activation_function itself."""
  if type(f) == str:
    f = act_mapper.get(f, None)
  return f


def is_act_func(f):
  """Registers <f> in the act_mapper."""
  name = f.__name__
  global act_mapper
  if name in act_mapper:
    logging.info("Already have a function named %s in act_mapper! Aborting", name)
  else:
    act_mapper[name] = f
  return f


def non_flatten_act_func(f):
  """Registers <f> in the after mapper, and in addition to that,
  makes it ignore the 'flatten' step."""
  def func(*args):
    if len(args) < 2 or args[1] not in ["flatten"]:
      return f(*args)
  func.__name__ = f.__name__
  return is_act_func(func)


@non_flatten_act_func
def relu(*args):
  return Functional(nn.functional.relu)



@non_flatten_act_func
def sgnlog(*args):
  return Functional(sgnlog_func)

@non_flatten_act_func
def ez_et_sgnlog(arch, name):
  c = arch.sizes[name]
  return Zoomer(Tracker(sgnlog(), c), c)

@non_flatten_act_func
def np_ez_sgnlog(arch, name):
  c = arch.sizes[name]
  f = sgnlog()
  return NegPoser(Zoomer(f, c), Zoomer(f, c))

@non_flatten_act_func
def np_ez_et_sgnlog(arch, name):
  c = arch.sizes[name]
  f = Tracker(sgnlog(), c)
  return NegPoser(Zoomer(f, c), Zoomer(f, c))



@non_flatten_act_func
def tanh(*args):
  return Functional(nn.functional.tanh)



@non_flatten_act_func
def p_sgnlog(arch, name):
  c = arch.sizes[name]
  return ParameterizedSgnlog(c)


#### STEP FUNCS ########################################################

step_mapper = {}

def str_to_step(f):
  """If <f> is a string, return the corresponding entry from step_mapper.
  Otherwise, assume <f> is a step_func itself."""
  if type(f) == str:
    f = step_mapper[f]
  return f


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


#### CREATION OF NET ###################################################

from .general import scaler_grad


net_mapper = {}

def str_to_net(f):
  """If <f> is a string, return the corresponding entry from net_mapper.
  Otherwise, assume <f> is a net itself."""
  if type(f) == str:
    f = net_mapper[f]
  return f


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
  
  def __call__(
    self, norm1_func = None, act_func = None, norm2_func = None,
    step_func = step
  ):
    """Returns a constructed net."""
    pipeline = []
    for name in self.names:
      mod0 = step_func(self, name)
      if mod0:
        pipeline.append((name, mod0))
      
      # Do not use normalizations after the last step.
      if name in self.next_names:
        
        # If at start, skip norm1 and activation.
        if name != "start":
          if norm1_func:
            name1 = "norm1_{}".format(name)
            mod1 = norm1_func(self, name)
            if mod1:
              pipeline.append((name1, mod1))
          
          if act_func:
            name2 = "act_{}".format(name)
            mod2 = act_func(self, name)
            if mod2:
              pipeline.append((name2, mod2))
        
        if norm2_func:
          name3 = "norm2_{}".format(name)
          mod3 = norm2_func(self, name)
          if mod3:
            pipeline.append((name3, mod3))
    
    return nn.Sequential(OrderedDict(pipeline))


def init_weights(gain, weight_norm = False, prop_grad = None):
  """Returns a function, that initializes our net's parameters
  recursively, using a variant of the xavier initialization with the
  given gain <gain>."""
  def func(module):
    if type(module) in [nn.Linear, nn.Conv2d]:
      nn.init.xavier_normal_(module.weight, gain)
      
      if weight_norm:
        # Decouple the weight's size from its direction.
        nn.utils.weight_norm(module)
      
      elif prop_grad:
        # Make weights change proportionally to their size if desired.
        if prop_grad is True:
          module.weight.register_hook(scaler_grad(module.weight))
        else:
          module.weight.register_hook(scaler_grad(module.weight, prop_grad))
      
      if module.bias is not None:
        with torch.no_grad():
          module.bias.fill_(0)
  
  return func
