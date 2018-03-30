import torch
import os, logging
from lib.util import *


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


def experiment(f):
  """Wrapper that takes a function <f> and returns a 'proper' version
  of that function. <f> should take no arguments and return experiment
  data (a two-layer dict) and the trained model. The returned function
  will automatically append this data to the prior experiment data
  and return the model."""
  def res():
    logging.info("Experiment {}".format(f.__name__))
    exp_data, model = f()
    append_all(exp_data, [f.__module__.split('.')[-1], f.__name__])
    return model
  res.__name__ = f.__name__
  return res


def get_plotter(mod_names, td_type, metrics):
  """Returns a plotting function that takes two arguments:
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
  def res(exps, smoothing = 0):
    exps = [f.__name__ for f in exps]
    for data in td_type:
      for f in metrics:
        plot_arrays(context(txt(mod_names), exps, [data, f.__name__]), smoothing)
  return res
