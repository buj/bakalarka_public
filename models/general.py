import torch
from torch import nn
from torch.autograd import Variable


class MyModule(nn.Module):
  """Custom nn_module with enhanced capabilities that are missing in
  pytorch."""
  
  def __call__(self, x):
    # Converts <x> to Variable if necessary, because !@#%ing pytorch
    # can't do stuff when Tensors and Variables are mixed.
    if isinstance(x, torch.Tensor):
      old_train = self.training
      self.train(False)
      x = Variable(x)
      res = super().__call__(x).data
      self.train(old_train)
      return res
    return super().__call__(x)
