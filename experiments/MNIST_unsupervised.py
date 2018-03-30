import torch
from lib.testdata import AutoDataset, load_MNIST
from lib.training import Trainer
from lib.experiments import experiment, get_plotter
from lib.functional import mean_squared_error


#### BOILERPLATE #######################################################

train_data, val_data = map(AutoDataset, load_MNIST())

criterion = torch.nn.MSELoss()
metrics = [mean_squared_error]

name = "MNIST_unsupervised"
train_plotter = get_plotter(name, ["train"], metrics)
val_plotter = get_plotter(name, ["val"], metrics)


#### (exact) PCA #######################################################

from lib.models.unsupervised import PCA
from lib.training import validate


pca_model = None

def train_PCA(train_data = train_data):
  """Trains the PCA algorithm on <train_data>."""
  train_loader = torch.utils.data.DataLoader(train_data, 10000, False)
  return PCA(train_loader)


def gen_PCA(name, code_size):
  def func():
    if pca_model is None:
      pca_model = train_PCA(train_data)
    pca_model.set_code_size(code_size)
    train_loader = torch.utils.data.DataLoader(train_data, 10000, False)
    train_meter = validate(pca_model, train_loader, metrics)
    val_loader = torch.utils.data.DataLoader(val_data, 10000, False)
    val_meter = validate(pca_model, val_loader, metrics)
    res = {
      "train": {fname : [x] for fname, x in train_meter.items()},
      "val": {fname : [x] for fname, x in val_meter.items()}
    }
    return res, pca_model
  func.__name__ = name
  return experiment(func)
