import torch

from lib.util import *

import os, logging


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


dir_path = os.path.dirname(__file__)


#### ARRAY MANIPULATION ################################################

def append(array, location):
  """Appends <array> to the array stored in <location>. More precisely,
  the arrays are concatenated along the zeroth dimension."""
  assert len(location) >= 1, "While appending to: empty location provided"
  array = array.view(1, -1)
  
  folder = os.path.join(dir_path, *location[:-1])
  filename = location[-1] + ".pt"
  path = os.path.join(folder, filename)
  try:
    old = torch.load(path)
    new = torch.cat((old, array), dim = 0)
  except IOError:
    logging.info("While appending to: file %s couldn't be read, creating anew", path)
    os.makedirs(folder, exist_ok = True)
    new = array
  torch.save(new, path)


def append_all(vals, exp_names):
  """
  Appends data in <vals> to the experiment's prior data.
  <vals>: a two-level dict containing experiment data.
  <exp_names>: The experiment's names.
  """
  for name1, sub in vals.items():
    for name2, x in sub.items():
      append(torch.Tensor(x), exp_names + [name1, name2])


#### EXPERIMENT STRUCTURE ##############################################

def save_params(params, exp_names):
  """Saves experiment parameters into experiment folder for later use."""
  folder = os.path.join(dir_path, *exp_names)
  filename = "params.txt"
  path = os.path.join(folder, filename)
  os.makedirs(folder, exist_ok = True)
  with open(path, "w") as fout:
    for name, value in params.items():
      if name == "kwargs":
        for name2, value2 in value.items():
          print(name2, "=", value2, file = fout)
      else:
        print(name, "=", value, file = fout)


class Experiment:
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
    if "prefix" not in self.params:
      self.params["prefix"] = func.__module__.split('.')[-1]
    save_params(self.params, [self.prefix, self.net, self.name])
  
  def __getattr__(self, name):
    """If we have no attribute named <name>, check the self.params dict."""
    if name in self.__dict__:
      return self.__dict__[name]
    params = self.__dict__["params"]
    if name in params:
      return params[name]
    raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))
  
  def __call__(self):
    """Carry out the experiment, and automatically store its data
    in the experiment's folder."""
    logging.info("Experiment {}".format(self.name))
    exp_data, model = self.func()
    names = [self.prefix, self.net, self.name]
    append_all(exp_data, names)
    save_model(model, names)
    return model


#### PLOTTING UTILS ####################################################

"""Current working directory. Is appended after the module name prefix."""
prefix = []
suffix = []


def plot_exps(exps, smoothing = 0, prefix = prefix, suffix = suffix, **kwargs):
  """Plots the given experiments. Prepends their names with the
  current working directory 'prefix', and after that come suffixes from
  'suffix'."""
  plot_arrays(context(prefix, exps, suffix), smoothing, **kwargs)


def get_plotter(mod_names, td_type, metrics):
  """
  Deprecated, no guarantees it works anymore. Use 'plot_exps' instead.
  
  Returns a plotting function that takes two arguments:
  <exps>: the experiments functions to be plotted.
  <smoothing>: how much emphasis should the plotter place on prior
    values. Default is 0.
  
  'get_plotter' itself takes three arguments:
  <mod_name>: Name of the experiment module. Determines the first level
    of the folder hierarchy.
  <td_type>: Whether we want to plot training or validation data. Is
    a list of these options.
  <metrics>: Which metrics we want the plotter to plot.
  """
  def res(exps, smoothing = 0, **kwargs):
    exps = [f.split('.') if type(f) is str else [f.net, f.name] for f in exps]
    for data in td_type:
      for f in metrics:
        plot_arrays(context(txt(mod_names) + cwd, exps, [data, f.__name__]), smoothing, **kwargs)
  return res


#### EXPERIMENT GENERATION #############################################

from torch import nn

from lib.functional import cross_entropy, accuracy
from lib.models.creation import str_to_step, str_to_norm, str_to_act, str_to_net, init_weights
from lib.training import Trainer


def gen_gen(
  train_data, val_data,
  criterion = nn.CrossEntropyLoss(), metrics = [cross_entropy, accuracy]
):
  """Returns a function that generates experiments on the given data."""
  trainer = Trainer(train_data, val_data, criterion, metrics, 512, torch.optim.SGD, num_epochs = 60)
  
  def gen(
    lr, net = "convnet1",
    step_func = "step", norm1_func = None, act_func = "relu", norm2_func = None,
    gain = nn.init.calculate_gain("relu"),
    weight_norm = False, prop_grad = None,
    parallel = False,
    name = "temp", **kwargs
  ):
    """Generates an experiment. The activation function is determined by
    <act_func>. Use of normalization is determined by <norm1_func>
    (before activation) and <norm2_func> (after activation)."""
    params = locals()
    
    # Translate from english names to python objects.
    net = str_to_net(net)
    step_func = str_to_step(step_func)
    norm1_func = str_to_norm(norm1_func)
    act_func = str_to_act(act_func)
    norm2_func = str_to_norm(norm2_func)
    
    def func():
      model = net(norm1_func, act_func, norm2_func)
      logging.info("Model info:\n%s", model)
      model.apply(init_weights(gain, weight_norm, prop_grad))
      if parallel:
        model = torch.nn.DataParallel(model)
      trainer.set(model = model, lr = lr, **kwargs)
      return trainer.train(), model
    
    return Experiment(params, func)
  
  return gen
