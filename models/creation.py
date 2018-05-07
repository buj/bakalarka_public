import torch
from torch import nn

import logging


#### LAYER CONSTRUCTORS ################################################

######## BASIC STUFF ###################################################

def conv(*args, in_size, out_size, gain = 1, **kwargs):
  """Returns a 2d convolutional layer with Xavier initialized weights,
  rescaled by <gain>. Ignores <in_size> and <out_size>."""
  f = nn.Conv2d(*args, **kwargs)
  nn.init.xavier_normal_(f.weight, gain)
  with torch.no_grad():
    f.bias.fill_(0.0)
  return f


def pool(*args, in_size, out_size, **kwargs):
  """Returns a 2d max-pooling layer."""
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
    norm = batch_class(out_size, *bn_args, **bn_kwargs)
    
    # Initialize the norm's weight and bias.
    with torch.no_grad():
      if norm.weight is not None:
        norm.weight.fill_(1.0)
      if norm.bias is not None:
        norm.bias.fill_(0.0)
    
    return nn.Sequential(f, norm)
  
  res.__name__ = func.__name__ + "_bn"
  return res


def layer_normed(func, *ln_args, after = True, **ln_kwargs):
  """
  Returns a layer constructor that wraps <func>: first, the layer
  <func> is executed. Then, layer normalization with parameters determined
  by <*ln_args> and <*ln_kwargs> is applied.
  """
  
  def res(*args, in_size, out_size, **kwargs):
    f = func(*args, in_size = in_size, out_size = out_size, **kwargs)
    norm = nn.LayerNorm(out_size, *ln_args, **ln_kwargs)
    return nn.Sequential(f, norm)
  
  res.__name__ = func.__name__ + "_ln"
  return res


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
  "dense": activated(dense, relu, nn.init.calculate_gain("relu"))
}


#### ARCHITECTURES #####################################################

from .general import Functional
from lib.functional import flatten


def mlp1(start, dense, **kwargs):
  """Returns a constructed multilayer perceptron whose exact architecture
  is described in the thesis. <start> is a function that returns a layer
  for preprocessing the input, <dense> is the linear layer constructor."""
  
  pipeline = [
    # Initial preprocessing of input.
    Functional(flatten),
    start(in_size = (3072,), out_size = (3072,)),
    
    # Many dense layers.
    dense(in_size = (3072,), out_size = (3000,)),
    *[dense(in_size = (3000 - 200*i,), out_size = (2800 - 200*i,)) for i in range(10)],
    *[dense(in_size = (1000 - 100*i,), out_size = (900 - 100*i,)) for i in range(9)],
    
    # Final dense layer.
    dense(in_size = (100,), out_size = (10,), last = True)
  ]
  
  pipeline = list(filter(lambda x: x, pipeline))
  return nn.Sequential(*pipeline)


def convnet2(start, conv, dense, **kwargs):
  """Returns a constructed convnet2. <start> is a function that returns
  a layer for preprocessing the input, <conv> is the convolution layer
  constructor, <dense> is the linear layer constructor."""
  
  pipeline = [
    # Initial preprocessing of input.
    start(in_size = (3, 32, 32), out_size = (3, 32, 32)),
    
    # First round of convolutions.
    conv(3, 6, 3, padding = 1, in_size = (3, 32, 32), out_size = (6, 32, 32)),
    conv(6, 12, 3, padding = 1, in_size = (6, 32, 32), out_size = (12, 32, 32)),
    conv(12, 12, 2, stride = 2, in_size = (12, 32, 32), out_size = (12, 16, 16)),
    
    # Second round of convolutions.
    conv(12, 24, 3, padding = 1, in_size = (12, 16, 16), out_size = (24, 16, 16)),
    conv(24, 48, 3, padding = 1, in_size = (24, 16, 16), out_size = (48, 16, 16)),
    conv(48, 48, 2, stride = 2, in_size = (48, 16, 16), out_size = (48, 8, 8)),
    
    # Last round of convolutions.
    conv(48, 96, 3, padding = 1, in_size = (48, 8, 8), out_size = (96, 8, 8)),
    conv(96, 192, 3, padding = 1, in_size = (96, 8, 8), out_size = (192, 8, 8)),
    conv(192, 192, 2, stride = 2, in_size = (192, 8, 8), out_size = (192, 4, 4)),
    
    # Flatten and dense.
    Functional(flatten),
    dense(in_size = (3072,), out_size = (200,)),
    dense(in_size = (200,), out_size = (10,), last = True)
  ]
  
  pipeline = list(filter(lambda x: x, pipeline))
  return nn.Sequential(*pipeline)


def all_convnet(start, conv, pool, **kwargs):
  """A convolutional network based on the 'All convolutional network'.
  <start> is a function that returns a layer for preprocessing the input,
  <conv> is the convolution layer constructor, <pool> is the pooling
  layer constructor."""
  
  pipeline = [
    # Initial preprocessing of input.
    start(in_size = (3, 32, 32), out_size = (3, 32, 32)),
    
    # First round of convolutions.
    conv(3, 96, 3, padding = 1, in_size = (3, 32, 32), out_size = (96, 32, 32)),
    conv(96, 96, 3, padding = 1, in_size = (96, 32, 32), out_size = (96, 32, 32)),
    conv(96, 96, 2, stride = 2, in_size = (96, 32, 32), out_size = (96, 16, 16)),
    
    # Second round of convolutions.
    conv(96, 192, 3, padding = 1, in_size = (96, 16, 16), out_size = (192, 16, 16)),
    conv(192, 192, 3, padding = 1, in_size = (192, 16, 16), out_size = (192, 16, 16)),
    conv(192, 192, 2, stride = 2, in_size = (192, 16, 16), out_size = (192, 8, 8)),
    
    # Last round of convolutions.
    conv(192, 192, 3, padding = 1, in_size = (192, 8, 8), out_size = (192, 8, 8)),
    conv(192, 192, 1, in_size = (192, 8, 8), out_size = (192, 8, 8)),
    conv(192, 10, 1, in_size = (192, 8, 8), out_size = (10, 8, 8), last = True),
    
    # Max pool to obtain results.
    pool(8, in_size = (10, 8, 8), out_size = (10, 1, 1)),
    Functional(flatten)
  ]
  
  pipeline = list(filter(lambda x: x, pipeline))
  return nn.Sequential(*pipeline)
