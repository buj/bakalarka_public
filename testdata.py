import torch
import torchvision
import os, logging


lib_dir = os.path.dirname(os.path.realpath(__file__))
testdata_dir = os.path.join(lib_dir, "testdata")


#### DATASET UTIL ######################################################

class AutoDataset(torch.utils.data.Dataset):
  """Datasets used for training autoencoders, that is, the input and
  targets are the same. Can be constructed from an arbitrary dataset
  by simply replacing its targets by its inputs."""
  
  def __init__(self, sub):
    super().__init__()
    self.sub = sub
  
  def __len__(self):
    return len(self.sub)
  
  def __getitem__(self, i):
    ins, _ = self.sub[i]
    return ins, ins


def data_means(data, k = 0):
  """Calculates the means for each attribute. Data should be an iterable
  containing the rows (or a batch of rows) of the data, and each row
  is a <k>-tuple (normally a 2-tuple, where the first element is the
  input and the second is the target). <k> determines which of these
  elements' means is calculated."""
  total = None
  count = 0
  for batch in data:
    batch = batch[k]
    value = torch.sum(batch, dim = 0)
    if total is None:
      total = value
    else:
      total += value
    count += batch.size()[0]
  total /= count
  return total


def data_stds(data, means = None, k = 0):
  """Calculates the variance for each attribute, in a similar fashion
  to 'data_means'."""
  if means is None:
    means = data_means(data)
  total = None
  count = 0
  for batch in data:
    batch = batch[k]
    value = torch.sum((batch - means)**2, dim = 0)
    if total is None:
      total = value
    else:
      total += value
    count += batch.size()[0]
  total /= count
  return total


def moments(data, k = 0):
  """Returns the first two moments (means and variances) for each attribute."""
  means = data_means(data, k)
  stds = data_stds(data, means, k)
  return means, stds


#### OTHER UTIL ########################################################

def to_categorical(y, num_classes):
  """1-hot encodes a tensor."""
  return torch.eye(num_classes)[y]


#### DATASET LOADERS ###################################################

def load_MNIST():
  path = os.path.join(testdata_dir, "MNIST")
  transform = torchvision.transforms.ToTensor()
  trainset = torchvision.datasets.MNIST(root = path, train = True, download = False, transform = transform)
  testset = torchvision.datasets.MNIST(root = path, train = False, download = False, transform = transform)
  return trainset, testset


def load_CIFAR10():
  path = os.path.join(testdata_dir, "CIFAR10")
  transform = torchvision.transforms.ToTensor()
  trainset = torchvision.datasets.CIFAR10(root = path, train = True, download = False, transform = transform)
  testset = torchvision.datasets.CIFAR10(root = path, train = False, download = False, transform = transform)
  return trainset, testset
