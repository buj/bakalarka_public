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


######## ALL CONVOLUTIONAL NETWORK #####################################

seeds = [3876642501229889088, 2934885003992402658, 2522141754085148986, 5858535661043479957]

def relu_():
  """Plain relu network."""
  for seed in seeds:
    g = gen(0.01, all_convnet, layers = relu(base), parallel = True, name = "relu 0.01")
    g(seed).train(10).save()

def relu_lsh_lsc_():
  """Network with layer shift and layer scale after each activation."""
  for seed in seeds:
    g = gen(0.01, all_convnet, layers = relu(lsh(lsc(base))), parallel = True, name = "relu lsh_lsc 0.01")
    g(seed).train(10).save()

def bn_relu_():
  """Batch normed (before activation) network."""
  for seed in seeds:
    g = gen(0.04, all_convnet, layers = bn(relu(base)), parallel = True, name = "bn relu 0.04")
    g(seed).train(10).save()


def drop_relu_():
  """Relu with dropout after first and second round of convolutions."""
  for seed in seeds:
    g = gen(0.01, all_convnet, layers = drop(relu(base)), parallel = True, name = "drop relu 0.01")
    g(seed).train(40).save()

def drop_relu_lsh_lsc_():
  """Layer shifted and layer scaled network, with dropout."""
  for seed in seeds:
    g = gen(0.01, all_convnet, layers = drop(relu(lsh(lsc(base)))), parallel = True, name = "drop relu lsh_lsc 0.01")
    g(seed).train(40).save()


######## MLP1 ##########################################################
