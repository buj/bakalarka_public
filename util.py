import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
import os, logging


dir_path = os.path.dirname(__file__)

#### PLOTTING UTILS ####################################################
######## ARRAY PLOTTING ####

def smooth(vals, k):
  """Smoothens the array <vals>. Returns a new array where the i-th value
  is the running average of the prior values and that value."""
  vals = torch.Tensor(vals)
  norms = torch.ones_like(vals)
  for i in range(vals.size()[0]):
    vals[i] += vals[i-1] * k
    norms[i] += norms[i-1] * k
  vals /= norms
  return vals


def load_arrays(names):
  """Load data from multiple experiments. <names> contains a list
  of experiment locations."""
  exp_folder = "experiments"
  res = []
  for name in names:
    assert len(name) >= 1, "While loading arrays: empty name"
    folder = os.path.join(dir_path, exp_folder, *name[:-1])
    filename = name[-1] + ".pt"
    path = os.path.join(folder, filename)
    try:
      arr = torch.load(path)
    except IOError:
      logging.info("While loading arrays: file %s can't be read from (does it exist?), skipping", path)
      continue
    res.append(torch.mean(arr, dim = 0))
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
  arrays = load_arrays(names)
  for name, arr in zip(names, arrays):
    y = smooth(arr, smoothing)
    y = torch.mean(y, dim = 0).numpy()
    caption = name[1]
    
    if stretch:
      x = np.linspace(0, 1, arr.shape[0])
    else:
      x = np.arange(arr.shape[0])
    
    num_points = arr.size()[0]
    func = plt.plot
    if num_points == 1:
      func = plt.scatter
    
    func(x, y, label = caption)
  
  plt.legend()
  plt.show()


######## IMAGE PLOTTING ####

def plot_img(tensor):
  """Plots a single image represented by <tensor>. The tensor should
  have dimensions (num_channels, height, width)."""
  plt.imshow(tensor.permute(1, 2, 0).squeeze(-1).numpy(), cmap = "gray")
  plt.show()


def plot_imgs(tensor_list, num_cols = 2, normalize = False):
  """Plots all tensors in <tensor_list> as images in a grid."""
  plot_img(torchvision.utils.make_grid(tensor_list, nrow = num_cols, normalize = normalize))


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


#### CONVENIENCE METHODS ###############################################

def txt(name):
  """Create a name-list from the given string. The character separating
  different names is the '.' symbol."""
  return name.split('.')


def context(left, names, right):
  """Creates a list of locations where each location
  begins with the list <left>, then something, and ends with the
  list <right>, where something comes from the list <names>."""
  res = [left + [x] + right for x in names]
  return res
