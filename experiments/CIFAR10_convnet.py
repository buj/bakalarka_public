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

from lib.models.creation import str_to_norm, str_to_act, str_to_net, init_weights
import lib.models.architectures


def gen(
  lr, net = "convnet1",
  norm1_func = None, act_func = "relu", norm2_func = None,
  gain = nn.init.calculate_gain("relu"),
  weight_norm = False, prop_grad = False,
  parallel = False,
  name = "temp", **kwargs
):
  """Generates an experiment. The activation function is determined by
  <act_func>. Use of normalization is determined by <norm1_func>
  (before activation) and <norm2_func> (after activation)."""
  params = locals()
  
  # Translate from english names to python objects.
  net = str_to_net(net)
  norm1_func = str_to_norm(norm1_func)
  act_func = str_to_act(act_func)
  norm2_func = str_to_norm(norm2_func)
  
  def func():
    model = net(norm1_func, act_func, norm2_func)
    model.apply(init_weights(gain, weight_norm, prop_grad))
    if parallel:
      model = torch.nn.DataParallel(model)
    trainer.set(model = model, lr = lr, **kwargs)
    return trainer.train(), model
  
  return Experiment(params, func)
