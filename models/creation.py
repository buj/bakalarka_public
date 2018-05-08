import torch
from torch import nn

import logging


#### LAYER CONSTRUCTORS ################################################

######## META STUFF ####################################################

def simple_wrap(wrapper, abbr):
  """Returns a function, that returns a layer constructor that can
  wrap a function with <wrapper>. It is assumed that <wrapper> takes
  <out_size> as the first positional argument."""
  
  def res1(func, *wp_args, **wp_kwargs):
    """Returns a layer constructor, that wraps <func> in <wrapper>."""
    
    def res2(*args, in_size, out_size, **kwargs):
      f = func(*args, in_size = in_size, out_size = out_size, **kwargs)
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
    f.bias.fill_(0.0)
  return f


def pool(*args, in_size, out_size, **kwargs):
  """Returns a 2d max-pooling layer. Ignores <in_size> and <out_size>."""
  return nn.MaxPool2d(*args, **kwargs)


def dense(in_size, out_size, gain = 1):
  """Returns a linear layer with weights initialized with Xavier and
  rescaled by <gain>, with input size <in_size> and output size <out_size>."""
  assert len(in_size) == 1, "Input to dense layer must be flat"
  assert len(out_size) == 1, "Output from dense layer must be flat"
  f = nn.Linear(*in_size, *out_size)
  nn.init.xavier_normal_(f.weight, gain)
  with torch.no_grad():
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
    f = func(*args, in_size = in_size, out_size = out_size, **kwargs)
    
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


layer_normed = simple_wrap(nn.LayerNorm, "ln")


#### SCALING AND SHIFTING LAYERS #######################################

from .general import Scaler, Shifter


element_scaled = simple_wrap(Scaler, "es")
element_shifted = simple_wrap(Shifter, "esh")

def np_es_wrapper(out_size, *args, **kwargs):
  """Returns the negposed elementwise scaler."""
  return NegPoser(Scaler(out_size, *args, **kwargs),
                  Scaler(out_size, *args, **kwargs))

negpos_scaled = simple_wrap(np_es_wrapper, "np_es")


def channeled(wrapper):
  """Useful for per-channel operations."""
  def res(out_size, *args, **kwargs):
    # Set all dimensions except the first to size 1.
    out_info = [*out_size]
    for i in range(1, len(out_size)):
      out_info[i] = 1
    return wrapper(out_info, *args, **kwargs)
  return res

channel_scaled = simple_wrap(channeled(Scaler), "cs")
channel_shifted = simple_wrap(channeled(Shifter), "csh")


#### ACTIVATIONS (wrap stuff) ##########################################

from lib.functional import sgnlog as sgnlog_func


def activated(func, act, gain = 1):
  """Returns a layer constructor that wraps <func>: the returned layer
  first performs <func> and then the activation act."""
  
  def res(*args, in_size, out_size, last = False, **kwargs):
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

identity = simple_act(lambda x: x)
identity.__name__ = "identity"


#### DEFAULTS ##########################################################

default_layers = {
  "start": identity,
  "conv": activated(conv, relu, nn.init.calculate_gain("relu")),
  "pool": pool,
  "dense": activated(dense, relu, nn.init.calculate_gain("relu")),
  "dropout": dropout
}


#### ARCHITECTURES #####################################################

from .general import Functional
from lib.functional import flatten


def mlp1(start, dense, dropout, **kwargs):
  """Returns a constructed multilayer perceptron whose exact architecture
  is described in the thesis. <start> is a function that returns a layer
  for preprocessing the input, <dense> is the linear layer constructor."""
  
  pipeline = [
    # Initial preprocessing of input.
    dropout(0.2, in_size = (3, 32, 32), out_size = (3, 32, 32)),
    Functional(flatten),
    start(in_size = (3072,), out_size = (3072,)),
    
    dense(in_size = (3072,), out_size = (3000,)),
    dropout(0.5, in_size = (3000,), out_size = (3000,)),
  ]
  # Many dense layers. Dropout after each dense layer.
  for i in range(10):
    pipeline.extend([
      dense(in_size = (3000 - 200*i,), out_size = (2800 - 200*i,)),
      dropout(0.5, in_size = (2800 - 200*i,), out_size = (2800 - 200*i,))
    ])
  for i in range(9):
    pipeline.extend([
      dense(in_size = (1000 - 100*i,), out_size = (900 - 100*i,)),
      dropout(0.5, in_size = (900 - 100*i,), out_size = (900 - 100*i,))
    ])
  
  # Final dense layer.
  pipeline.append(dense(in_size = (100,), out_size = (10,), last = True))
  
  pipeline = list(filter(lambda x: x, pipeline))
  return nn.Sequential(*pipeline)


def convnet2(start, conv, dense, dropout, **kwargs):
  """Returns a constructed convnet2. <start> is a function that returns
  a layer for preprocessing the input, <conv> is the convolution layer
  constructor, <dense> is the linear layer constructor."""
  
  pipeline = [
    # Initial preprocessing of input.
    dropout(0.2, in_size = (3, 32, 32), out_size = (3, 32, 32)),
    start(in_size = (3, 32, 32), out_size = (3, 32, 32)),
    
    # First round of convolutions.
    conv(3, padding = 1, in_size = (3, 32, 32), out_size = (6, 32, 32)),
    conv(3, padding = 1, in_size = (6, 32, 32), out_size = (12, 32, 32)),
    conv(2, stride = 2, in_size = (12, 32, 32), out_size = (12, 16, 16)),
    
    # Second round of convolutions.
    conv(3, padding = 1, in_size = (12, 16, 16), out_size = (24, 16, 16)),
    conv(3, padding = 1, in_size = (24, 16, 16), out_size = (48, 16, 16)),
    conv(2, stride = 2, in_size = (48, 16, 16), out_size = (48, 8, 8)),
    dropout(0.5, in_size = (48, 8, 8), out_size = (48, 8, 8)),
    
    # Last round of convolutions.
    conv(3, padding = 1, in_size = (48, 8, 8), out_size = (96, 8, 8)),
    conv(3, padding = 1, in_size = (96, 8, 8), out_size = (192, 8, 8)),
    conv(2, stride = 2, in_size = (192, 8, 8), out_size = (192, 4, 4)),
    dropout(0.5, in_size = (192, 4, 4), out_size = (192, 4, 4)),
    
    # Flatten and dense.
    Functional(flatten),
    dense(in_size = (3072,), out_size = (200,)),
    dropout(0.5, in_size = (200,), out_size = (200,)),
    dense(in_size = (200,), out_size = (10,), last = True)
  ]
  
  pipeline = list(filter(lambda x: x, pipeline))
  return nn.Sequential(*pipeline)


def all_convnet(start, conv, pool, dropout, **kwargs):
  """A convolutional network based on the 'All convolutional network'.
  <start> is a function that returns a layer for preprocessing the input,
  <conv> is the convolution layer constructor, <pool> is the pooling
  layer constructor."""
  
  pipeline = [
    # Initial preprocessing of input.
    dropout(0.2, in_size = (3, 32, 32), out_size = (3, 32, 32)),
    start(in_size = (3, 32, 32), out_size = (3, 32, 32)),
    
    # First round of convolutions.
    conv(3, padding = 1, in_size = (3, 32, 32), out_size = (96, 32, 32)),
    conv(3, padding = 1, in_size = (96, 32, 32), out_size = (96, 32, 32)),
    conv(2, stride = 2, in_size = (96, 32, 32), out_size = (96, 16, 16)),
    dropout(0.5, in_size = (96, 16, 16), out_size = (96, 16, 16)),
    
    # Second round of convolutions.
    conv(3, padding = 1, in_size = (96, 16, 16), out_size = (192, 16, 16)),
    conv(3, padding = 1, in_size = (192, 16, 16), out_size = (192, 16, 16)),
    conv(2, stride = 2, in_size = (192, 16, 16), out_size = (192, 8, 8)),
    dropout(0.5, in_size = (192, 8, 8), out_size = (192, 8, 8)),
    
    # Last round of convolutions.
    conv(3, padding = 1, in_size = (192, 8, 8), out_size = (192, 8, 8)),
    conv(1, in_size = (192, 8, 8), out_size = (192, 8, 8)),
    conv(1, in_size = (192, 8, 8), out_size = (10, 8, 8), last = True),
    
    # Max pool to obtain results.
    pool(8, in_size = (10, 8, 8), out_size = (10, 1, 1)),
    Functional(flatten)
  ]
  
  pipeline = list(filter(lambda x: x, pipeline))
  return nn.Sequential(*pipeline)
