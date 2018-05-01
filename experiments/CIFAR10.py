import torch

from lib.testdata import load_CIFAR10
from . import Experiment, get_plotter, gen_gen, prefix, suffix, plot_exps
import lib.models.architectures 

import logging


#### BOILERPLATE #######################################################

train_data, val_data = load_CIFAR10()
gen = gen_gen(train_data, val_data)

prefix = ["CIFAR10", "convnet1"]
suffix = ["train", "cross_entropy"]

# Other, less used things to be plotted.
t_acc = ["train", "accuracy"]
v_cent = ["val", "cross_entropy"]
v_acc = ["val", "accuracy"]

"""
Deprecated. Use only if very lazy, and 'plot_exps' doesn't suffice
(but it totally should).

name = "CIFAR10"
train_plotter = get_plotter(name, ["train"], metrics)
val_plotter = get_plotter(name, ["val"], metrics)
"""

#### EXPERIMENTS #######################################################
