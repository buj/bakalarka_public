import torch

from . import *

import logging


#### BOILERPLATE #######################################################

from lib.testdata import load_CIFAR10
from lib.functional import cross_entropy, accuracy
from lib.util import prefix, prefix_as


ctx.train_data, ctx.val_data = load_CIFAR10()
ctx.criterion = torch.nn.CrossEntropyLoss()
ctx.metrics = [cross_entropy, accuracy]

ctx.bind_trainer()


prefix[:] = ["experiments", "CIFAR10"]
suffix[:] = t_cent


#### EXPERIMENTS #######################################################

from lib.models.creation import *


######## MLP1 experiments ##############################################

############ IoLinear ##################################################

io_layers = {**default_layers}
io_layers["dense"] = activated(io_dense, relu, nn.init.calculate_gain("relu"))

ios = [
  gen(0.02 * 10**-i, mlp1, num_epochs = 30,
    layers = io_layers, name = "io_dense{}".format(i)
  ) for i in range(5)
]
