import torch
from torch import nn

import logging


#### LAYER CONSTRUCTORS ################################################

######## BASIC STUFF ###################################################

def conv(bias = True):
  """Returns a convolution layer constructor. <bias> determines whether
  the layer should have the bias parameter or not. Weights are not
  initialized, biases are set to 0."""
  def res(*args, **kwargs):
    f = nn.Conv2d(*args, bias = bias, **kwargs)
    with torch.no_grad():
      if f.bias is not None:
        f.bias.fill_(0.0)
    return f
  return res


def dense(bias = True):
  """Returns a dense layer constructor. <bias> determines whether the
  layer should have the bias parameter or not. Weights are not
  initialized, biases are set to 0."""
  def res(*args, **kwargs):
    f = nn.Linear(*args, bias = bias, **kwargs)
    with torch.no_grad():
      if f.bias is not None:
        f.bias.fill_(0.0)
    return f
  return res


def dropout(p, dims):
  """Returns a dropout layer. The probability of dropping a node is
  set to <p>."""
  if dims == "1d":
    return nn.Dropout(p)
  elif dims == "2d":
    return nn.Dropout2d(p)
  return nn.Dropout3d(p)


######## ADVANCED BASIC STUFF ##########################################

from .general import IoLinear


def io_dense(in_size, out_size, gain = 1):
  """Returns an io_linear layer. Ordinary weights are left uninitialized,
  input weights and output weights are set to 0. Input size is <in_size>,
  output size is <out_size>."""
  f = IoLinear(in_size, out_size)
  with torch.no_grad():
    f.bias.fill_(0.0)
  return f


def bias_lr(k, func):
  """Wraps the layer <func> so that its bias learns <k>-times faster."""
  def res(*args, **kwargs):
    f = func(*args, **kwargs)
    if f.bias is not None:
      f.bias.register_hook(lambda grad: grad * k)
    return f
  return res


def xavier_init(gain, func):
  """(Re)initializes the dense/conv layer constructed from <func>
  using Xavier initialization with gain <gain>."""
  def res(*args, **kwargs):
    f = func(*args, **kwargs)
    nn.init.xavier_normal_(f.weight, gain)
    return f
  return res

# Default dense and conv layers.
dfl_dense = xavier_init(1, dense())
dfl_conv = xavier_init(1, conv())


def scale_w(k, func):
  """Rescales the weights of a dense/conv layer constructed from <func>."""
  def res(*args, **kwargs):
    f = func(*args, **kwargs)
    with torch.no_grad():
      f.weight *= k
    return f
  return res


######## META STUFF ####################################################

def simple_wrap(wrapper, abbr):
  """Returns a function, that returns a layer constructor that can
  wrap a function with <wrapper>."""
  
  def res1(func, *wp_args, **wp_kwargs):
    """Returns a layer constructor, that wraps <func> in <wrapper>."""
    
    def res2(size):
      wp = wrapper(size, *wp_args, **wp_kwargs)
      f = func(size)
      
      # Returns a list of modules, in the order in which they are to
      # be executed.
      res = [wp]
      if type(f) is list:
        res.extend(f)
      else:
        res.append(f)
      return res
    
    res2.__name__ = "{}_{}".format(abbr, func.__name__) if func is not None else abbr
    return res2
  
  return res1


######## NORM LAYERS (wrap stuff) ######################################

def batch_norm(func, *bn_args, **bn_kwargs):
  """
  Returns a layer constructor that wraps <func>: first, batch norm
  is executed and then <func>. The batch norm's parameters are deterimned
  by <*bn_args> and <*bn_kwargs>.
  """
  
  def res(size):
    if len(size) <= 2:
      batch_class = nn.BatchNorm1d
    elif len(size) == 3:
      batch_class = nn.BatchNorm2d
    else:
      batch_class = nn.BatchNorm3d
    norm = batch_class(size[0], *bn_args, **bn_kwargs)
    
    # Initialize the norm's weight and bias.
    with torch.no_grad():
      if norm.weight is not None:
        norm.weight.fill_(1.0)
      if norm.bias is not None:
        norm.bias.fill_(0.0)
    
    f = func(size)
    
    # Returns a list of modules, in the order in which they are to
    # be executed.
    res = [norm]
    if type(f) is list:
      res.extend(f)
    else:
      res.append(f)
    return res
  
  res.__name__ = "bn_{}".format(func.__name__) if func is not None else "bn"
  return res


layer_norm = simple_wrap(nn.LayerNorm, "ln")


#### SCALING AND SHIFTING LAYERS #######################################

from .general import Scaler, Shifter, NegPoser


def pScaler(size):
  """Returns a Scaler whose 'proportional learning' has been enabled."""
  return Scaler(size, eps = 10**-6)


scale = simple_wrap(Scaler, "sc")
pscale = simple_wrap(pScaler, "psc")
shift = simple_wrap(Shifter, "sh")


def channeled(wrapper):
  """Useful for per-channel operations."""
  def res(size):
    # Set all dimensions except the first to size 1.
    subsize = [*size]
    for i in range(1, len(size)):
      subsize[i] = 1
    return wrapper(subsize)
  return res

channel_scale = simple_wrap(channeled(Scaler), "csc")
channel_pscale = simple_wrap(channeled(pScaler), "cpsc")
channel_shift = simple_wrap(channeled(Shifter), "csh")


def np_sc_wrapper(size):
  """Returns the negposed scaler."""
  return NegPoser(Scaler(size), Scaler(size))

negpos_scale = simple_wrap(channeled(np_sc_wrapper), "np_sc")


def layered(wrapper):
  """Useful for per-layer operations."""
  def res(size):
    # Ignore <out_size>, have only 1 weight per entire layer.
    return wrapper((1,))
  return res

layer_scale = simple_wrap(layered(Scaler), "ls")
layer_pscale = simple_wrap(layered(pScaler), "lps")
layer_shift = simple_wrap(layered(Shifter), "lsh")


#### ACTIVATIONS (wrap stuff) ##########################################

from lib.functional import sgnlog as sgnlog_func
from lib.models.general import Functional


def simple_act(f, name = None):
  """Wraps a function to make it into a 'simple activation function
  constructor'. It ignores all the arguments passed by the net constructor."""
  if name is not None:
    f.__name__ = name
  
  def res(*args, **kwargs):
    return Functional(f)
  
  res.__name__ = f.__name__
  return res


relu_act = simple_act(nn.functional.relu)
tanh_act = simple_act(nn.functional.tanh)
sgnlog_act = simple_act(sgnlog_func)

expu_act = simple_act(lambda x: torch.exp(x) - 1, "exp")

identity = simple_act(lambda x: x, "identity")
