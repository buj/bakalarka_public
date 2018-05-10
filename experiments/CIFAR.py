from torch import nn
from lib.models.general import Functional
from lib.functional import flatten


#### ARCHITECTURES for the CIFAR datasets ##############################

def mlp1(out_size):
  """Returns a function that returns a mlp1 model, with output size <out_size>."""
  
  def res(start, dense, dropout, **kwargs):
    """Returns a constructed multilayer perceptron for CIFAR10 or CIFAR100
    classification (determined by <out_size>). <start> is a function that
    returns a layer for preprocessing the input, <dense> is the linear
    layer constructor."""
    
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
    pipeline.append(dense(in_size = (100,), out_size = (out_size,), last = True))
    
    pipeline = list(filter(lambda x: x, pipeline))
    return nn.Sequential(*pipeline)
  
  res.__name__ = "mlp1"
  return res



def convnet2(out_size):
  """Returns a function that returns a convnet2 architecture, with
  output size <out_size>."""
  
  def res(start, conv, dense, dropout, **kwargs):
    """Returns a constructed convnet2 for CIFAR10 or CIFAR100, based on
    <out_size>. <start> is a function that returns a layer for
    preprocessing the input, <conv> is the convolution layer
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
      dense(in_size = (200,), out_size = (out_size,), last = True)
    ]
    
    pipeline = list(filter(lambda x: x, pipeline))
    return nn.Sequential(*pipeline)
  
  res.__name__ = "convnet2"
  return res



def all_convnet(out_size):
  """Returns a function that constructs the all_convnet, with output
  size <out_size>."""
  
  def res(start, conv, dropout, **kwargs):
    """A convolutional network based on the 'All convolutional network',
    for CIFAR10 or CIFAR100. The size of output is determined by <out_size>.
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
      conv(3, padding = 1, stride = 2, in_size = (96, 32, 32), out_size = (96, 16, 16)),
      dropout(0.5, in_size = (96, 16, 16), out_size = (96, 16, 16)),
      
      # Second round of convolutions.
      conv(3, padding = 1, in_size = (96, 16, 16), out_size = (192, 16, 16)),
      conv(3, padding = 1, in_size = (192, 16, 16), out_size = (192, 16, 16)),
      conv(3, padding = 1, stride = 2, in_size = (192, 16, 16), out_size = (192, 8, 8)),
      dropout(0.5, in_size = (192, 8, 8), out_size = (192, 8, 8)),
      
      # Last round of convolutions.
      conv(3, in_size = (192, 8, 8), out_size = (192, 8, 8)),
      conv(1, in_size = (192, 6, 6), out_size = (192, 6, 6)),
      conv(1, in_size = (192, 6, 6), out_size = (out_size, 6, 6), last = True),
      
      # Avg pool to obtain results.
      nn.AvgPool2d(6),
      Functional(flatten)
    ]
    
    pipeline = list(filter(lambda x: x, pipeline))
    return nn.Sequential(*pipeline)
  
  res.__name__ = "all_convnet"
  return res



def small_net(out_size):
  """Returns a function that constructs a convnet similar to LeNet, with
  output size <out_size>."""
  
  def res(start, conv, dense, **kwargs):
    """A small convolutional network, for testing that doesn't
    require a GPU. Inspired by LeNet."""
    
    pipeline = [
      # Initial preprocessing of input.
      start(in_size = (3, 32, 32), out_size = (3, 32, 32)),
      
      # First round of convolutions.
      conv(5, in_size = (3, 32, 32), out_size = (6, 28, 28)),
      nn.MaxPool2d(2),
      
      # Second round of convolutions.
      conv(5, in_size = (6, 14, 14), out_size = (16, 10, 10)),
      nn.MaxPool2d(2),
      
      # Flatten and dense.
      Functional(flatten),
      dense(in_size = (400,), out_size = (120,)),
      dense(in_size = (120,), out_size = (84,)),
      dense(in_size = (84,), out_size = (out_size,), last = True)
    ]
    
    pipeline = list(filter(lambda x: x, pipeline))
    return nn.Sequential(*pipeline)
  
  res.__name__ = "small_net"
  return res
