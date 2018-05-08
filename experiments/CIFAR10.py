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


############ Io ########################################################

io_layers = {**default_layers,
  "dense": activated(io_dense, relu, nn.init.calculate_gain("relu"))
}


############ Sgnlog ####################################################

sgnlog_layers = {**default_layers,
  "conv": activated(conv, sgnlog),
  "dense": activated(dense, sgnlog)
}

"""bn_sgnlog_0.0007"""
bn_sgnlog_layers = {**default_layers,
  "start": batch_normed(identity),
  "conv": activated(batch_normed(conv), sgnlog),
  "dense": activated(batch_normed(dense), sgnlog)
}


############ ReLU ######################################################

relu_gain = nn.init.calculate_gain("relu")

bn_relu_layers = {**default_layers,
  "start": batch_normed(identity),
  "conv": activated(batch_normed(conv), relu, relu_gain),
  "dense": activated(batch_normed(dense), relu, relu_gain)
}

in_relu_layers = {**default_layers,
  "start": element_scaled(element_shifted(identity)),
  "conv": channel_scaled(channel_shifted(activated(conv, relu, relu_gain))),
  "dense": element_scaled(element_shifted(activated(dense, relu, relu_gain)))
}
