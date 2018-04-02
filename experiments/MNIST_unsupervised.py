import torch

from lib.testdata import AutoDataset, load_MNIST
from lib.training import Trainer
from lib.experiments import experiment, get_plotter
from lib.functional import randn_init, mean_squared_error


#### BOILERPLATE #######################################################

train_data, val_data = map(AutoDataset, load_MNIST())

criterion = torch.nn.MSELoss()
metrics = [mean_squared_error]

name = "MNIST_unsupervised"
train_plotter = get_plotter(name, ["train"], metrics)
val_plotter = get_plotter(name, ["val"], metrics)

trainer = Trainer(train_data, val_data, criterion, metrics, 100, torch.optim.SGD)


#### (exact) PCA #######################################################

from lib.models.unsupervised import PCA
from lib.training import validate


pca_model = None

def train_PCA(train_data = train_data):
  """Trains the PCA algorithm on <train_data>."""
  train_loader = torch.utils.data.DataLoader(train_data, 10000, False)
  return PCA(train_loader)


def gen_PCA(code_size, name = "temp"):
  def func():
    global pca_model
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


pcas = [gen_PCA(c, "pca{}".format(c)) for c in [1, 100, 200, 300, 400, 500, 600, 700]]


#### LinearAutoencoder ####

from lib.models.unsupervised import LinearAutoencoder

def gen_lin(code_size, lr, scale = 1, name = "temp"):
  def func():
    model = LinearAutoencoder(784, code_size)
    randn_init(scale)(model)
    trainer.set(model = model, lr = lr)
    return trainer.train(), model
  func.__name__ = name
  return experiment(func)

lins = [gen_lin(100, 4 * 2**-i, name = "lin100_{}".format(i)) for i in range(2)]


#### PCALikeAutoencoder ####

from lib.models.unsupervised import PCALikeAutoencoder




#### PointAutoencoder ####

from lib.models.unsupervised import PointAutoencoder

def gen_pt(code_size, lr, scale = 1, adjust_sharp = True, name = "temp"):
  def func():
    model = PointAutoencoder(784, code_size, adjust_sharp = adjust_sharp)
    randn_init(scale)(model)
    trainer.set(model = model, lr = lr)
    return trainer.train(), model
  func.__name__ = name
  return experiment(func)

pts = [gen_pt(100, 64 * 2**-i, name = "pt100_{}".format(i)) for i in range(2)]
pts_soft = [gen_pt(100, 64 * 2**-i, False, name = "pt_soft100_{}".format(i)) for i in range(2)]
