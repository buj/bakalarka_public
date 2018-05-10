import torch

from . import *

import logging


#### BOILERPLATE #######################################################

from lib.testdata import normed_CIFAR100
from lib.functional import cross_entropy, accuracy
from lib.util import prefix, prefix_as


ctx.train_data, ctx.val_data = normed_CIFAR100()
ctx.criterion = torch.nn.CrossEntropyLoss()
ctx.metrics = [cross_entropy, accuracy]


prefix[:] = ["experiments", "CIFAR100"]
suffix[:] = t_cent


#### ARCHITECTURES #####################################################

from .CIFAR import *

mlp1 = mlp1(100)
convnet2 = convnet2(100)
all_convnet = all_convnet(100)
small_net = small_net(100)


#### EXPERIMENTS #######################################################

from lib.models.constructors import *
