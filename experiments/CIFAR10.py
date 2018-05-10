import torch

from . import *

import logging


#### BOILERPLATE #######################################################

from lib.testdata import normed_CIFAR10
from lib.functional import cross_entropy, accuracy
from lib.util import prefix, prefix_as


ctx.train_data, ctx.val_data = normed_CIFAR10()
ctx.criterion = torch.nn.CrossEntropyLoss()
ctx.metrics = [cross_entropy, accuracy]


prefix[:] = ["experiments", "CIFAR10"]
suffix[:] = t_cent


#### ARCHITECTURES #####################################################

from .CIFAR import *

mlp1 = mlp1(10)
convnet2 = convnet2(10)
all_convnet = all_convnet(10)
small_net = small_net(10)


#### EXPERIMENTS #######################################################

from lib.models.constructors import *
