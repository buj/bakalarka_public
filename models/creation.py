import torch
from torch import nn

import logging


#### LAYER CONSTRUCTORS ################################################

######## META STUFF ####################################################

def simple_wrap(wrapper, abbr, last_ok = False):
  """Returns a function, that returns a layer constructor that can
  wrap a function with <wrapper>. It is assumed that <wrapper> takes
  <out_size> as the first positional argument."""
  
  def res1(func, *wp_args, **wp_kwargs):
    """Returns a layer constructor, that wraps <func> in <wrapper>."""
    
    def res2(*args, in_size, out_size, **kwargs):
      f = func(*args, in_size = in_size, out_size = out_size, last = last, **kwargs)
      if kwargs.get("last", False) and not last_ok:
        # In the last layer, ignore processing that comes after the activation.
        # This is implemented with the 'last' flag.
        return f
      wp = wrapper(out_size, *wp_args, **wp_kwargs)
      return nn.Sequential(f, wp)
    
    res2.__name__ = func.__name__ + "_" + abbr
    return res2
  
  return res1


######## BASIC STUFF ###################################################

def conv(*args, in_size, out_size, gain = 1, **kwargs):
  """Returns a 2d convolutional layer with Xavier initialized weights,
  rescaled by <gain>."""
  f = nn.Conv2d(in_size[0], out_size[0], *args, **kwargs)
  nn.init.xavier_normal_(f.weight, gain)
  with torch.no_grad():
    if f.bias is not None:
      f.bias.fill_(0.0)
  return f


def avg_pool(*args, in_size, out_size, **kwargs):
  """Returns an avg-pooling layer."""
  if len(in_size) <= 2:
    pool_class = nn.AvgPool1d
  elif len(in_size) == 3:
    pool_class = nn.AvgPool2d
  else:
    pool_class = nn.AvgPool3d
  return pool_class(*args, **kwargs)


def max_pool(*args, in_size, out_size, **kwargs):
  """Returns a max-pooling layer."""
  if len(in_size) <= 2:
    pool_class = nn.MaxPool1d
  elif len(in_size) == 3:
    pool_class = nn.MaxPool2d
  else:
    pool_class = nn.MaxPool3d
  return pool_class(*args, **kwargs)


def dense(*args, in_size, out_size, gain = 1, **kwargs):
  """Returns a linear layer with weights initialized with Xavier and
  rescaled by <gain>, with input size <in_size> and output size <out_size>."""
  assert len(in_size) == 1, "Input to dense layer must be flat"
  assert len(out_size) == 1, "Output from dense layer must be flat"
  f = nn.Linear(*in_size, *out_size, *args, **kwargs)
  nn.init.xavier_normal_(f.weight, gain)
  with torch.no_grad():
    if f.bias is not None:
      f.bias.fill_(0.0)
  return f


def dropout(p, in_size, out_size):
  """Returns a dropout layer. The probability of dropping a node is
  set to <p>."""
  if len(in_size) <= 2:
    drop_class = nn.Dropout1d
  elif len(in_size) == 3:
    drop_class = nn.Dropout2d
  else:
    drop_class = nn.Dropout3d
  return drop_class(p)


######## ADVANCED BASIC STUFF ##########################################

from .general import IoLinear


def io_dense(in_size, out_size, gain = 1):
  """Returns an io_linear layer with standard input-to-output weights
  initialized with Xavier and rescaled by <gain>. Input weights and
  output weights are set to 0. Input size is <in_size>, output size
  is <out_size>."""
  assert len(in_size) == 1, "Input to io_dense layer must be flat"
  assert len(out_size) == 1, "Output from io_dense layer must be flat"
  f = IoLinear(*in_size, *out_size)
  nn.init.xavier_normal_(f.weight, gain)
  with torch.no_grad():
    f.bias.fill_(0.0)
  return f


######## NORM LAYERS (wrap stuff) ######################################

def batch_normed(func, *bn_args, **bn_kwargs):
  """
  Returns a layer constructor that wraps <func>: first, the layer
  <func> is executed. Then, batch normalization with parameters determined
  by <*bn_args> and <*bn_kwargs> is applied.
  """
  
  def res(*args, in_size, out_size, **kwargs):
    # No point having bias prior to batch norm.
    if "bias" not in kwargs:
      kwargs["bias"] = False
    f = func(*args, in_size = in_size, out_size = out_size, **kwargs)
    if kwargs.get("last", False):
      # In the last layer, ignore processing that comes after the activation.
      return f
    
    if len(in_size) <= 2:
      batch_class = nn.BatchNorm1d
    elif len(in_size) == 3:
      batch_class = nn.BatchNorm2d
    else:
      batch_class = nn.BatchNorm3d
    norm = batch_class(out_size[0], *bn_args, **bn_kwargs)
    
    # Initialize the norm's weight and bias.
    with torch.no_grad():
      if norm.weight is not None:
        norm.weight.fill_(1.0)
      if norm.bias is not None:
        norm.bias.fill_(0.0)
    
    return nn.Sequential(f, norm)
  
  res.__name__ = func.__name__ + "_bn"
  return res


layer_normed = simple_wrap(nn.LayerNorm, "ln", True)


#### SCALING AND SHIFTING LAYERS #######################################

from .general import Scaler, Shifter


def pScaler(*args, **kwargs):
  """Returns a Scaler whose 'proportional learning' has been enabled."""
  return Scaler(*args, eps = 10**-6, **kwargs)


element_scaled = simple_wrap(Scaler, "es", True)
element_pscaled = simple_wrap(pScaler, "eps", True)
element_shifted = simple_wrap(Shifter, "esh")

def np_es_wrapper(out_size, *args, **kwargs):
  """Returns the negposed elementwise scaler."""
  return NegPoser(Scaler(out_size, *args, **kwargs),
                  Scaler(out_size, *args, **kwargs))

negpos_scaled = simple_wrap(np_es_wrapper, "np_es", True)


def channeled(wrapper):
  """Useful for per-channel operations."""
  def res(out_size, *args, **kwargs):
    # Set all dimensions except the first to size 1.
    out_info = [*out_size]
    for i in range(1, len(out_size)):
      out_info[i] = 1
    return wrapper(out_info, *args, **kwargs)
  return res

channel_scaled = simple_wrap(channeled(Scaler), "cs", True)
channel_pscaled = simple_wrap(channeled(pScaler), "cps", True)
channel_shifted = simple_wrap(channeled(Shifter), "csh")


def layered(wrapper):
  """Useful for per-layer operations."""
  def res(out_size, *args, **kwargs):
    # Ignore <out_size>, have only 1 weight per entire layer.
    return wrapper((1,), *args, **kwargs)
  return res

layer_scaled = simple_wrap(layered(Scaler), "ls", True)
layer_pscaled = simple_wrap(layered(pScaler), "lps", True)
layer_shifted = simple_wrap(layered(Shifter), "lsh", True)


#### ACTIVATIONS (wrap stuff) ##########################################

from lib.functional import sgnlog as sgnlog_func
from lib.models.general import Functional


def activated(func, act, gain = 1):
  """Returns a layer constructor that wraps <func>: the returned layer
  first performs <func> and then the activation act."""
  
  def res(*args, in_size, out_size, last = False, **kwargs):
    # Activation layer is the only layer that doesn't propagate the 'last' flag.
    # It consumes it, thus enabling layers before the activation to be constructed.
    f = func(*args, in_size = in_size, out_size = out_size, gain = (1 if last else gain), **kwargs)
    if last:
      return f
    a = act(*args, in_size = in_size, out_size = out_size, **kwargs)
    return nn.Sequential(f, a)
  
  res.__name__ = func.__name__ + "_" + act.__name__
  return res


def simple_act(f):
  """Wraps a function to make it into a 'simple activation function
  constructor'. It ignores all the arguments passed by the net constructor."""
  def res(*args, **kwargs):
    return Functional(f)
  res.__name__ = f.__name__
  return res


relu = simple_act(nn.functional.relu)
tanh = simple_act(nn.functional.tanh)
sgnlog = simple_act(sgnlog_func)

expu = simple_act(lambda x: torch.exp(x) - 1)
expu.__name__ = "exp"

identity = simple_act(lambda x: x)
identity.__name__ = "identity"
