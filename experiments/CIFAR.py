from torch import nn
from lib.models.general import Functional
from lib.functional import flatten


#### UTILS #############################################################

def whole_act(size, kwargs):
  return [
    kwargs["before"](size),
    kwargs["act"](size),
    kwargs["after"](size)
  ]


def squash(pipeline):
  """Converts lists in the pipeline into something nn.Sequentials, and
  filters out Nones."""
  res = []
  for x in pipeline:
    if x is None:
      continue
    if type(x) is list:
      res.extend(x)
    else:
      res.append(x)
  return res


#### ARCHITECTURES for the CIFAR datasets ##############################

from lib.models.constructors import base

dfl_conv = base["conv"]
dfl_dense = base["dense"]


def mlp1(out_size):
  """Returns a function that returns a mlp1 model, with output size <out_size>."""
  
  def res(**kwargs):
    global dfl_dense
    dense = kwargs["dense"]
    before = kwargs["before"]
    act = kwargs["act"]
    after = kwargs["after"]
    dropout = kwargs["dropout"]
    
    pipeline = [
      # Initial preprocessing of input.
      Functional(flatten),
      after((3072,)),
      
      dense(3072, 3000),
      *whole_act((3000,), kwargs),
      dropout(0.1, "1d")
    ]
    # Many dense layers. Dropout after each dense layer.
    for i in range(10):
      ins = 3000 - 200*i
      outs = ins - 200
      pipeline.extend([
        dense(ins, outs),
        *whole_act((outs,), kwargs),
        dropout(0.1, "1d")
      ])
    for i in range(9):
      ins = 1000 - 100*i
      outs = ins - 100
      pipeline.extend([
        dense(ins, outs),
        *whole_act((outs,), kwargs),
        dropout(0.1, "1d")
      ])
    
    # Final dense layer, with gain 1.
    pipeline.extend([
      dfl_dense(100, out_size),
      before((out_size,))
    ])
    
    return nn.Sequential(*squash(pipeline))
  
  res.__name__ = "mlp1"
  return res



def convnet2(out_size):
  """Returns a function that returns a convnet2 architecture, with
  output size <out_size>."""
  
  def res(**kwargs):
    global dfl_dense
    conv = kwargs["conv"]
    dense = kwargs["dense"]
    before = kwargs["before"]
    act = kwargs["act"]
    after = kwargs["after"]
    dropout = kwargs["dropout"]
    
    pipeline = [
      # Initial preprocessing of input.
      after((3, 32, 32)),
      
      # First round of convolutions.
      conv(3, 6, 3, padding = 1),
      *whole_act((6, 32, 32), kwargs),
      conv(6, 12, 3, padding = 1),
      *whole_act((12, 32, 32), kwargs),
      conv(12, 12, 2, stride = 2),
      *whole_act((12, 16, 16), kwargs),
      
      # Second round of convolutions.
      conv(12, 24, 3, padding = 1),
      *whole_act((24, 16, 16), kwargs),
      conv(24, 48, 3, padding = 1),
      *whole_act((48, 16, 16), kwargs),
      conv(48, 48, 2, stride = 2),
      *whole_act((48, 8, 8), kwargs),
      
      # Last round of convolutions.
      conv(48, 96, 3, padding = 1),
      *whole_act((96, 8, 8), kwargs),
      conv(96, 192, 3, padding = 1),
      *whole_act((192, 8, 8), kwargs),
      conv(192, 192, 2, stride = 2),
      *whole_act((192, 4, 4), kwargs),
      dropout(0.5, "2d"),
      
      # Flatten and dense.
      Functional(flatten),
      dense(3072, 200),
      *whole_act((200,), kwargs),
      dropout(0.5, "2d"),
      
      # The last dense layer has gain 1.
      dfl_dense(200, out_size),
      before((out_size,))
    ]
    
    return nn.Sequential(*squash(pipeline))
  
  res.__name__ = "convnet2"
  return res



def all_convnet(out_size):
  """Returns a function that constructs the all_convnet, with output
  size <out_size>."""
  
  def res(**kwargs):
    """A convolutional network based on the 'All convolutional network'."""
    conv = kwargs["conv"]
    before = kwargs["before"]
    act = kwargs["act"]
    after = kwargs["after"]
    dropout = kwargs["dropout"]
    
    pipeline = [
      # Initial preprocessing of input.
      after((3, 32, 32)),
      
      # First round of convolutions.
      conv(3, 96, 3, padding = 1),
      *whole_act((96, 32, 32), kwargs),
      conv(96, 96, 3, padding = 1),
      *whole_act((96, 32, 32), kwargs),
      conv(96, 96, 3, padding = 1, stride = 2),
      *whole_act((96, 16, 16), kwargs),
      dropout(0.5, "2d"),
      
      # Second round of convolutions.
      conv(96, 192, 3, padding = 1),
      *whole_act((192, 16, 16), kwargs),
      conv(192, 192, 3, padding = 1),
      *whole_act((192, 16, 16), kwargs),
      conv(192, 192, 3, padding = 1, stride = 2),
      *whole_act((192, 8, 8), kwargs),
      dropout(0.5, "2d"),
      
      # Last round of convolutions.
      conv(192, 192, 3),
      *whole_act((192, 6, 6), kwargs),
      conv(192, 192, 1),
      *whole_act((192, 6, 6), kwargs),
      conv(192, out_size, 1),
      *whole_act((out_size, 1, 1), kwargs),
      
      # Avg pool to obtain results.
      nn.AvgPool2d(6),
      Functional(flatten)
    ]
    
    return nn.Sequential(*squash(pipeline))
  
  res.__name__ = "all_convnet"
  return res



def small_net(out_size):
  """Returns a function that constructs a convnet similar to LeNet, with
  output size <out_size>."""
  
  def res(**kwargs):
    """A small convolutional network, for testing that doesn't
    require a GPU. Inspired by LeNet."""
    global dfl_dense
    conv = kwargs["conv"]
    dense = kwargs["dense"]
    before = kwargs["before"]
    act = kwargs["act"]
    after = kwargs["after"]
    dropout = kwargs["dropout"]
    
    pipeline = [
      # Initial preprocessing of input.
      after((3, 32, 32)),
      
      # First round of convolutions.
      conv(3, 6, 5),
      *whole_act((6, 28, 28), kwargs),
      nn.MaxPool2d(2),
      
      # Second round of convolutions.
      conv(6, 16, 5),
      *whole_act((16, 10, 10), kwargs),
      nn.MaxPool2d(2),
      
      # Flatten and dense.
      Functional(flatten),
      dense(400, 120),
      *whole_act((120,), kwargs),
      dense(120, 84),
      *whole_act((84,), kwargs),
      
      # The last dense layer has gain 1.
      dfl_dense(84, out_size),
      before((84,))
    ]
    
    return nn.Sequential(*squash(pipeline))
  
  res.__name__ = "small_net"
  return res
