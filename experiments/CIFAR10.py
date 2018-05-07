import torch

from . import plot_exps

import logging


#### BOILERPLATE #######################################################

from lib.testdata import load_CIFAR10
from lib.util import prefix
from . import suffix, t_cent, t_acc, v_cent, v_acc


train_data, val_data = load_CIFAR10()

prefix = ["CIFAR10"]
suffix = t_cent


#### EXPERIMENTS #######################################################

from . import Experiment, gen
