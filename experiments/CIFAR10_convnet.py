import torch
from torch import nn

from lib.testdata import load_CIFAR10
from lib.training import Trainer
from lib.experiments import experiment, get_plotter
from lib.functional import cross_entropy


#### BOILERPLATE #######################################################

train_data, val_data = load_CIFAR10()

criterion = nn.CrossEntropyLoss()
metrics = [cross_entropy]

name = "CIFAR10_convnet"
train_plotter = get_plotter(name, ["train"], metrics)
val_plotter = get_plotter(name, ["val"], metrics)

trainer = Trainer(train_data, val_data, criterion, metrics, 128, torch.optim.SGD)


#### RELU with NOTHING #################################################

from lib.models.convnet import convnet, init_weights
from lib.models.general import act_after


def gen_relu0(lr, name = "temp"):
  """Generates an experiment with a plain relu convolutional network."""
  def func():
    model = convnet(act_after(nn.functional.relu))
    model.apply(init_weights(nn.init.calculate_gain("relu")))
    trainer.set(model = model, lr = lr)
    return trainer.train(), model
  func.__name__ = name
  return experiment(func)
