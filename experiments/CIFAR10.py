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

from lib.models.creation import mlp1, convnet2, all_convnet
