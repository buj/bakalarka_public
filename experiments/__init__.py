import torch

from lib.util import to_path
import os, copy

import logging


"""
Conventions:
- Experiment's name is implicitly set as the module's name for convenience.
- An experiment's data is a two-level dict. The first level
  determines whether these are training or validation data, the second
  level determines the measured metric (like accuracy, mean squared error, ...).
- The data are stored in the experiment's folder. Each experiment's data
  should be stored in separate folders. There can be a nested folder
  structure. The exact location is determined by the experiment's names:
  a list of strings, each string representing a folder nested in the
  previous folder.
- Paths to files are given as lists of strings. Each string except the
  last one corresponds to a folder on the path, and the last string
  is the name of the file.
"""


#### PLOTTING ##########################################################

from contextlib import contextmanager
from lib.util import context, plot_arrays


"""<suffix> determines the suffix of all paths: whether we are looking
for training or validation data, and what metric we want to see."""
suffix = []

@contextmanager
def suffix_as(val):
  """With this context manager, we can temporarily set <suffix> to
  a specified value."""
  global suffix
  old_suffix = copy.copy(suffix)
  suffix[:] = val
  yield
  suffix[:] = old_suffix


# Commonly used suffixes.
t_cent = ["train", "cross_entropy"]
t_acc = ["train", "accuracy"]
v_cent = ["val", "cross_entropy"]
v_acc = ["val", "accuracy"]


def plot_exps(exps, smoothing = 0, **kwargs):
  """Plots the given experiments. Prepends their names with the
  current working directory 'prefix', and after that come suffixes from
  'suffix'."""
  global suffix
  xlabel = "steps"
  ylabel = "training"
  if suffix[0] == "val":
    xlabel = "epochs"
    ylabel = "validation"
  ylabel += " " + suffix[1].replace('_', ' ')
  plot_arrays(context([], exps, suffix), smoothing, xlabel = xlabel, ylabel = ylabel, **kwargs)


#### ARRAY MANIPULATION ################################################

def _append(array, names):
  """Appends <array> to the array stored in location determined by
  <names>. The arrays are concatenated along the zeroth dimension."""
  assert len(names) >= 1, "While appending to: empty location provided"
  array = array.view(1, -1)
  
  folder = names[:-1]
  filename = names[-1] + ".pt"
  path = to_path(*folder, filename)
  try:
    old = torch.load(path)
    new = torch.cat((old, array), dim = 0)
  except IOError:
    logging.info("While appending to: file %s couldn't be read, creating anew", path)
    os.makedirs(os.path.dirname(path), exist_ok = True)
    new = array
  torch.save(new, path)


def _append_all(vals, exp_names):
  """
  Appends data in <vals> to the experiment's prior data.
  <vals>: a two-level dict containing experiment data.
  <exp_names>: The experiment's names.
  """
  for name1, sub in vals.items():
    for name2, x in sub.items():
      _append(torch.Tensor(x), exp_names + [name1, name2])


#### GLOBAL VARS #######################################################

class ExpContext:
  """Encapsulates variables so that they can be shared and modified
  in other modules.  Because %@!!?!#& python can't do that."""
  
  def __init__(self):
    self.train_data = None
    self.val_data = None
    self.criterion = None
    self.metrics = None


"""The one and only one context."""
ctx = ExpContext()


#### EXPERIMENT STRUCTURE ##############################################

from lib.util import prefix, prefix_as, save_model
from lib.training import Trainer


def _save_params(params, exp_names):
  """Saves experiment parameters into experiment folder for later use."""
  filename = "params.txt"
  path = to_path(*exp_names, filename)
  os.makedirs(os.path.dirname(path), exist_ok = True)
  with open(path, "w") as fout:
    for name, value in params.items():
      if name == "kwargs":
        for name2, value2 in value.items():
          print(name2, "=", value2, file = fout)
      else:
        print(name, "=", value, file = fout)


def _save_seed(seed, exp_names):
  """Appends <seed> to the file with seeds used to generate nets."""
  if seed is not None:
    filename = "seeds.txt"
    path = to_path(*exp_names, filename)
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, "a") as fout:
      print(seed, file = fout)


class ExpParams:
  """A helper class that contains experiment parameters."""
  
  def __init__(
    self, lr, net, layers,
    parallel = False,
    name = "temp", **kwargs
  ):
    """Just store the provided arguments."""
    self.lr = lr
    self.net = net
    self.layers = layers
    self.parallel = parallel
    self.name = name
    self.kwargs = kwargs
    
    # Store the prefix. This is where the experiment folder will be located.
    global prefix
    self.prefix = copy.copy(prefix)
    if len(self.prefix) == 0 or self.prefix[-1] != net.__name__:
      self.prefix.append(net.__name__)
  
  def info(self):
    """Returns an informational dict."""
    return {
      "lr": self.lr, "net": self.net.__name__,
      **{name: f.__name__ for name, f in self.layers.items()},
      "name": self.name, **self.kwargs
    }


class Experiment:
  """Implements an experiment."""
  
  def __init__(self, p, seed = None):
    """<p> contains the learning rate for the optimizer, the rough
    network architecture <p.net>, and the details of the network
    are determined by <p.layers>. Random seed is set to <seed>."""
    if seed is not None:
      torch.manual_seed(seed)
      logging.info("Random seed is: %d\n", torch.initial_seed())
    
    # Construct the model.
    self.model = p.net(**p.layers)
    logging.info("Model info:\n%s", self.model)
    if p.parallel:
      self.model = torch.nn.DataParallel(self.model)
    
    # Create a trainer for the model.
    global ctx
    self.trainer = Trainer(
      self.model, ctx.train_data, ctx.val_data,
      ctx.criterion, ctx.metrics, torch.optim.SGD
    )
    self.trainer.set(lr = p.lr, **p.kwargs)
    
    self.seed = seed
    self.P = p
  
  def train(self, num_epochs = 1):
    """Proxy to self.trainer's train method."""
    self.trainer.train(num_epochs)
    return self
  
  def save(self):
    """Store the experiment data in the experiment's folder."""
    exp_data = self.trainer.data()
    with prefix_as(self.P.prefix):
      _append_all(exp_data, [self.P.name])
      _save_params(self.P.info(), [self.P.name])
      _save_seed(self.seed, [self.P.name])
      save_model(self.model, [self.P.name])


def gen(*args, **kwargs):
  """A convenient way for generating experiments."""
  p = ExpParams(*args, **kwargs)
  def func(seed = None):
    res = Experiment(p, seed)
    return res
  return func
