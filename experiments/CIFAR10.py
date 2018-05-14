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
mlp2 = mlp2(10)
convnet2 = convnet2(10)
all_convnet = all_convnet(10)
small_net = small_net(10)


#### EXPERIMENTS #######################################################

from lib.models.constructors import *


######## ALL CONVOLUTIONAL NETWORK #####################################

seeds = [403670034471441514, 24169147534048556, 195976647745127160, 928871599459262954]

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

seeds2 = [453678291562578089, 115490442701496540, 34641834943987341, 262608601143284066]
