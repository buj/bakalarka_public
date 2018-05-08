import torch

import logging


#### CONVENIENCE METHODS ###############################################

import os, copy
from contextlib import contextmanager


_dir_path = os.path.dirname(__file__)

"""<prefix> determines the prefix of all paths."""
prefix = []

@contextmanager
def prefix_as(val):
  """With this context manager, we can temporarily set <prefix> to
  a specified value."""
  global prefix
  old_prefix = copy.copy(prefix)
  prefix[:] = val
  yield
  prefix[:] = old_prefix


def to_path(*names):
  """Converts a list of nested folder names into a 'path'."""
  global prefix
  tokens = []
  for x in names:
    tokens.extend(x.strip().split())
  return os.path.join(_dir_path, *prefix, *tokens)


def txt(name):
  """Create a name-list from the given string."""
  return name.strip().split()


def context(left, names, right):
  """Creates a list of locations where each location
  begins with the list <left>, then something, and ends with the
  list <right>, where something comes from the list <names>."""
  res = [left + x + right if type(x) is list else left + [x] + right for x in names]
  return res


#### PLOTTING UTILS ####################################################

from matplotlib import pyplot as plt


######## ARRAY PLOTTING ################################################

def _smooth(vals, k):
  """Smoothens the array <vals>. Returns a new array where the i-th value
  is the running average of the prior values and that value."""
  vals = torch.Tensor(vals)
  norms = torch.ones_like(vals)
  for i in range(1, vals.size()[0]):
    vals[i] += vals[i-1] * k
    norms[i] += norms[i-1] * k
  vals /= norms
  return vals


def _load_arrays(exps):
  """Load data from multiple experiments. <names> contains a list
  of experiment locations."""
  res = []
  for exp_names in exps:
    assert len(exp_names) >= 1, "While loading arrays: empty name"
    folder = exp_names[:-1]
    filename = exp_names[-1] + ".pt"
    path = to_path(*folder, filename)
    try:
      arr = torch.load(path)
      res.append(torch.mean(arr, dim = 0))
    except IOError:
      logging.info("While loading arrays: file %s can't be read from (does it exist?), skipping", path)
      res.append(None)
      continue
  return res


def plot_arrays(names, smoothing = 0, stretch = False):
  """
  Plots data from multiple experiments.
  <names>: List of experiment locations.
  <smoothing>: How much weight should be put on previously measured values.
  <stretch>: If experiments have various lengths (i.e. numbers of epochs
    run), this option will adjust their lengths so that they have equal
    length.
  """
  arrays = _load_arrays(names)
  
  # Do smoothing, legend and plot.
  for name, arr in zip(names, arrays):
    if arr is None:
      continue
    if arr.shape[0] == 1:
      arr = torch.cat((arr, arr), dim = 0)
    
    y = _smooth(arr, smoothing)
    if stretch:
      x = torch.linspace(0, 1, arr.shape[0])
    else:
      x = torch.arange(arr.shape[0])
    
    plt.plot(x.numpy(), y.numpy(), label = name[0])
  
  plt.legend()
  plt.show()


######## IMAGE PLOTTING ################################################

from torchvision.utils import make_grid


def plot_img(tensor):
  """Plots a single image represented by <tensor>. The tensor should
  have dimensions (num_channels, height, width)."""
  plt.imshow(tensor.permute(1, 2, 0).squeeze(-1).numpy(), cmap = "gray")
  plt.show()


def plot_imgs(tensor_list, num_cols = 2, normalize = False):
  """Plots all tensors in <tensor_list> as images in a grid."""
  plot_img(make_grid(tensor_list, nrow = num_cols, normalize = normalize))


def plot_auto(model, data, indices = range(8)):
  """Shows <model>'s reconstructions of a few samples from <data>. That,
  which samples are shown, is determined by <indices>."""
  to_show = []
  for i in indices:
    img = data[i][0]
    out = model(img)
    diff = img - out
    to_show.extend([img, out, diff])
  plot_imgs(to_show, num_cols = 3, normalize = True)


######## DENSITY PLOTTING ##############################################

def plot_hist(data, bins = 30, **kwargs):
  """Plots the histogram of data and shows it."""
  plt.hist(data.detach(), bins, **kwargs)
  plt.show()


#### LOADING STORED STUFF ##############################################

import dill


def save_model(model, names):
  """Saves a model to the location 'experiments/*names'."""
  if type(names) is str:
    names = txt(names)
  
  path = to_path(*names, "model")
  try:
    torch.save(model, path, pickle_module = dill)
  except:
    with open(path, "wb") as fout:
      dill.dump(model, fout)


def load_model(names):
  """Loads a model from the location 'experiments/*names'."""
  if type(names) is str:
    names = txt(names)
  
  path = to_path(*names, "model")
  try:
    model = torch.load(path, pickle_module = dill)
  except:
    with open(path, "rb") as fin:
      model = dill.load(fin)
  return model


#### RANDOM STUFF ######################################################

def num_params(model, only_train = True):
  """Returns the number of parameters in the model. If <only_trainable>
  is true, only trainable parameters are included."""
  return sum(x.numel() for x in model.parameters() if x.requires_grad or not only_train)

def params(model, only_train = True):
  """Returns the generator of model's (trainable) parameters."""
  return (x for x in model.parameters() if x.requires_grad or not only_train)

def glue(tensors):
  """Concatenates all the tensors into a single one-dimensional vector."""
  return torch.cat([x.view(-1) for x in tensors])


def std(x, center = None):
  """Returns the distance from <center> to the given tensor <x>. If
  <center> is None, it is assumed to be the mean of <x>."""
  if center is None:
    center = torch.mean(x)
  return torch.mean((x - center)**2)**0.5


def params_named(name, model):
  """Returns a generator of all the model's parameters named <name>.
  For example, it can be used to obtain all the weights ot a model,
  excluding biases. It can be useful if it can be assumed that the
  parameters named 'weight' have the standard meaning of multiplying
  something."""
  res = []
  def func(module):
    # Beware, some code very dependent on pytorch.
    if "_parameters" in module.__dict__:
      _parameters = module.__dict__["_parameters"]
      if name in _parameters:
        res.append(_parameters[name])
  model.apply(func)
  return res
