import torch
from torch import nn

from lib.testdata import load_CIFAR10
from lib.functional import cross_entropy, accuracy
from . import Experiment, get_plotter, gen_gen

import logging


#### BOILERPLATE #######################################################

train_data, val_data = load_CIFAR10()
gen = gen_gen(train_data, val_data)

criterion = nn.CrossEntropyLoss()
metrics = [cross_entropy, accuracy]

name = "CIFAR10"
train_plotter = get_plotter(name, ["train"], metrics)
val_plotter = get_plotter(name, ["val"], metrics)


#### EXPERIMENTS #######################################################
