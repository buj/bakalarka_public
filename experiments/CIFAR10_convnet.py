import torch
from torch import nn

from lib.testdata import load_CIFAR10
from lib.training import Trainer
from lib.experiments import experiment, get_plotter
from lib.functional import cross_entropy, accuracy


#### BOILERPLATE #######################################################

train_data, val_data = load_CIFAR10()

criterion = nn.CrossEntropyLoss()
metrics = [cross_entropy, accuracy]

name = "CIFAR10_convnet"
train_plotter = get_plotter(name, ["train"], metrics)
val_plotter = get_plotter(name, ["val"], metrics)

trainer = Trainer(train_data, val_data, criterion, metrics, 512, torch.optim.SGD, num_epochs = 12)


#### RELU with NOTHING #################################################

from lib.models.convnet import step, layer_normed, batch_normed, convnet, init_weights
from lib.models.general import act_after


def gen(
  lr, parallel = False,
  after_func = act_after(nn.functional.relu),
  step_func = step, gain = nn.init.calculate_gain("relu"),
  weight_norm = False,
  name = "temp", **kwargs
):
  """Generates an experiment. The activation function is determined by
  <after_func>. Use of normalization is determined by <step_func>."""
  def func():
    model = convnet(after_func, step_func)
    model.apply(init_weights(gain, weight_norm))
    if parallel:
      model = torch.nn.DataParallel(model)
    trainer.set(model = model, lr = lr, **kwargs)
    return trainer.train(), model
  func.__name__ = name
  return experiment(func)
