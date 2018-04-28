import torch
from torch import nn

from lib.testdata import load_CIFAR10
from lib.training import Trainer
from lib.experiments import Experiment, get_plotter
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

from lib.models.creation import step_mapper, after_mapper, net_mapper, init_weights
import lib.models.architectures


def gen(
  lr, convnet = "convnet1", after_func = "relu", step_func = "step",
  gain = nn.init.calculate_gain("relu"),
  weight_norm = False,
  parallel = False,
  name = "temp", **kwargs
):
  """Generates an experiment. The activation function is determined by
  <after_func>. Use of normalization is determined by <step_func>."""
  params = locals()
  
  # Translate from english names to python objects.
  convnet = net_mapper[convnet]
  after_func = after_mapper.get(after_func, None)
  step_func = step_mapper.get(step_func, None)
  
  def func():
    model = convnet(after_func, step_func)
    model.apply(init_weights(gain, weight_norm))
    if parallel:
      model = torch.nn.DataParallel(model)
    trainer.set(model = model, lr = lr, **kwargs)
    return trainer.train(), model
  
  return Experiment(params, func)
