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
  plot_arrays(context([], exps, suffix), smoothing, **kwargs)


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


#### EXPERIMENT STRUCTURE ##############################################

from lib.util import prefix, prefix_as, save_model


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


class _Experiment:
  """Implements an experimental setup. Can be called to carry out
  the experiment coded in it."""
  
  def __init__(self, params, func):
    """
    <params>: the parameters of the experiment (for further
      reproduction)
    <func>: the function that is to be carried out. It should take
      no arguments and return a tuple containing the experiment data
      and trained model.
    """
    self.params = params
    self.func = func
    if "name" not in self.params:
      self.params["name"] = func.__name__
    
    # Store the prefix. This is where the experiment folder will be located.
    global prefix
    if "prefix" not in self.params:
      self.params["prefix"] = copy.copy(prefix)
      if len(prefix) == 0 or prefix[-1] != self.net:
        self.params["prefix"].append(self.net)
  
  def __getattr__(self, name):
    """If we have no attribute named <name>, check the self.params dict."""
    if name in self.__dict__:
      return self.__dict__[name]
    params = self.__dict__["params"]
    if name in params:
      return params[name]
    raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))
  
  def __call__(self, seed = None):
    """Carry out the experiment, and automatically store its data
    in the experiment's folder. If <seed> is not None, set the rng seed
    to this value."""
    if seed is not None:
      torch.manual_seed(seed)
      logging.info("Random seed is: %d\n", torch.initial_seed())
    
    logging.info("Experiment {}".format(self.name))
    exp_data, model = self.func()
    with prefix_as(self.prefix):
      _append_all(exp_data, [self.name])
      _save_params(self.params, [self.name])
      _save_seed(seed, [self.name])
      save_model(model, [self.name])
    return model


#### EXPERIMENT GENERATION #############################################

from lib.models.constructors import base
from lib.training import Trainer


class ExpContext:
  """Encapsulates variables so that they can be shared and modified
  in other modules.  Because %@!!?!#& python can't do that."""
  
  def __init__(self):
    self.train_data = None
    self.val_data = None
    self.metrics = None
    self.trainer = None
    self.layers = {**base}
  
  def bind_trainer(self):
    """Bind the trainer to train_data, val_data, criterion, ... Usually
    used only once, during initialization."""
    self.trainer = Trainer(self.train_data, self.val_data, self.criterion, self.metrics, torch.optim.SGD)


"""The one and only one context."""
ctx = ExpContext()


def gen(
  lr, net, layers = ctx.layers,
  parallel = False,
  name = "temp", **kwargs
):
  """Generates an experiment. The rough network architecture is
  determined by <net>, and the details are determined by <layers>."""  
  params = {
    "lr": lr, "net": net.__name__,
    **{name: f.__name__ for name, f in layers.items()},
    "name": name, **kwargs
  }
  
  def func():
    model = net(**layers)
    logging.info("Model info:\n%s", model)
    if parallel:
      model = torch.nn.DataParallel(model)
    
    global ctx
    ctx.trainer.set(model = model, lr = lr, **kwargs)
    return ctx.trainer.train(), model
  
  return _Experiment(params, func)
