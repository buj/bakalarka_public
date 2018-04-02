import numpy as np
import torch
from torch import nn

from lib.functional import flatten
from lib.testdata import moments
from lib.util import plot_imgs
from .general import MyModule

import logging


class PCA:
  """An implementation of the PCA algorithm, without neural nets."""
  
  def __init__(self, train_data, eps = 10**-12):
    logging.info("Training PCA...")
    
    # Normalize the training data.
    logging.info("Calculating means and variances...")
    self.means, self.stds = moments(train_data)
    self.stds += eps
    
    # Compute covariance matrix.
    logging.info("Computing covariance matrix...")
    cov = None
    milestone = 0.0
    for i, batch in enumerate(train_data):
      batch = batch[0]
      normalized = flatten((batch - self.means) / self.stds)
      gain = normalized.t().matmul(normalized)
      if cov is None:
        cov = gain
      else:
        cov += gain
      
      # Info logging.
      progress = (i+1) / len(train_data)
      if progress - milestone >= 0.1 or progress == 1.0:
        logging.info("%d%% done", int(100*progress))
    
    self.input_size = cov.size()[0]
    self.code_size = self.input_size
    
    # Compute the eigenvectors.
    _, self.W = map(torch.from_numpy, np.linalg.eigh(cov.numpy()))
    order = torch.arange(self.input_size - 1, -1, -1, out = torch.LongTensor())
    self.W = self.W[:, order]
  
  def set_code_size(self, c):
    assert 1 <= c <= self.input_size, "Given code_size is not in bounds: not(1 <= {} <= {})".format(c, self.input_size)
    self.code_size = c
    return self
  
  def code(self, x):
    x = flatten((x - self.means) / self.stds)
    return x.matmul(self.W[:, :self.code_size])
  
  def decode(self, y):
    return y.matmul(self.W[:, :self.code_size].t()) * self.stds.view(-1) + self.means.view(-1)
  
  def __call__(self, x):
    o_size = x.size()
    return self.decode(self.code(x)).view(*o_size)
  
  def plot_transition(self, img, indices = None):
    """Plots the reconstruction of <img> while increasing the code size."""
    if indices is None:
      indices = ((i * (self.input_size - 1)) // 8 for i in range(9))
    to_show = []
    for i in indices:
      out = self.set_code_size(i)(img)
      to_show.append(out)
    plot_imgs(to_show, num_cols = 8, normalize = True)


#### LINEAR AUTOENCODERS ###############################################

class LinearAutoencoder(MyModule):
  """Network that gets as input vector <x>, applies a linear transformation
  on it to obtain the code of <x>. To decode, apply a (different) linear
  transformation on the code. The input and output have the same dimensions,
  and the goal is to minimize the differences between input and output."""
  
  def __init__(self, input_size, code_size):
    super().__init__()
    self.coder = nn.Linear(input_size, code_size, True)
    self.decode = nn.Linear(code_size, input_size, True)
  
  def code(self, x):
    return self.coder(flatten(x))
  
  def forward(self, x):
    o_size = x.size()
    return self.decode(self.code(x)).view(*o_size)


class PCALikeAutoencoder(MyModule):
  """Similar to LinearAutoencoder, except that this network enforces
  that the two transformations are transposes of one another, and there
  is no bias."""
  
  def __init__(self, input_size, code_size):
    super().__init__()
    self.weights = nn.Parameter(torch.Tensor(input_size, code_size))
    self.normalized_weights = None
  
  def __setattr__(self, name, value):
    """Special check to see if we are changing the 'weights' variable.
    If yes, we reset 'normalized_weights' back to None."""
    if name == "weights":
      self.normalized_weights = None
    super().__setattr__(name, value)
  
  def code(self, x):
    if self.normalized_weights is None:
      self.normalized_weights = nn.functional.normalize(self.weights, dim = 0)
    return x.matmul(self.normalized_weights)
  
  def decode(self, y):
    if self.normalized_weights is None:
      self.normalized_weights = nn.functional.normalize(self.weights, dim = 0)
    return y.matmul(self.normalized_weights.t())
  
  def forward(self, x):
    return self.decode(self.code(x))


#### POINT AUTOENCODER #################################################

def pairwise_distance(x, points, p = 2, eps = 10**-12):
    """
    Returns the L_p distance function, tuned for use in PointAutoencoder.
    Distance functions should take 2 values:
    <x>: Input vector(s), first dimension is batch dimension and the second
      dimension contains the vector.
    <points>: The datapoints stored by PointAutoencoder. First dimension
      is which point, second dimension contains the vector.
    And they should return a tensor of shape (batch_size, num_points),
    each value representing closeness of the input vector to that
    particular point.
    """
    x = x.unsqueeze(1)
    points = points.unsqueeze(0)
    diff = torch.abs(x - points)
    return torch.sum((diff + eps)**p, dim = -1)**(1/p)


def adjust_sharpness_grad(model, grad_input, grad_output):
  """Should be called after 'backward()' on PointAutoencoder. Adjusts
  the gradient on the 'sharpness' parameter."""
  if model.sharpness.grad is not None:
    model.sharpness.grad.data *= model.sharpness.data
    if model.sharpness.grad.data[0] > model.sharpness.data[0]:
      model.sharpness.grad.data = model.sharpness.data


class PointAutoencoder(MyModule):
  """Special class of autoencoders. First, the input is compared to
  multiple 'datapoints'---vectors of the same length as input. Each of
  the datapoints is then assigned a weight, based on how close it was
  to the input. These weights add up to 1, and they form the code of
  the input. To decode, create a linear combination of datapoints,
  weighted by aforementioned weights."""
  
  def __init__(self, input_size, code_size, dist_func = pairwise_distance, adjust_sharp = True):
    super().__init__()
    self.points = nn.Parameter(torch.Tensor(code_size, input_size))
    self.dist_func = dist_func
    if adjust_sharp:
      self.sharpness = nn.Parameter(torch.ones(1))
      self.register_backward_hook(adjust_sharpness_grad)
    else:
      self.sharpness = 1
  
  def code(self, x):
    x = flatten(x)
    return nn.functional.softmin(self.sharpness * self.dist_func(x, self.points), dim = -1)
  
  def decode(self, y):
    return y.matmul(self.points)
  
  def forward(self, x):
    o_size = x.size()
    return self.decode(self.code(x)).view(*o_size)
