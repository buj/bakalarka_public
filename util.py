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
  return os.path.join(_dir_path, *prefix, *names)


def txt(name):
  """Create a name-list from the given string. The character separating
  different names is the '.' symbol."""
  return name.split('.')


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
