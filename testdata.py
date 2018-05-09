import torch

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


#### META DATASET LOADER ###############################################

from torchvision import transforms


def normed(f):
  """Returns a normalized dataset. <f> is the torchvision dataset getter."""
  name = f.__name__
  
  def res():
    logging.info("Loading normed %s dataset...", name)
    path = os.path.join(testdata_dir, name)
    
    # Normalize data.
    path_means = os.path.join(path, "means.pt")
    path_stds = os.path.join(path, "stds.pt")
    try:
      means = torch.load(path_means)
      stds = torch.load(path_stds)
    except Exception as e:
      # Probably not found, need to calculate it.
      trainset = f(root = path, train = True, download = True, transform = transforms.ToTensor())
      
      # Calculate means.
      means = torch.zeros_like(trainset[0][0])
      for ins, outs in trainset:
        means += ins
      means /= len(trainset)
      
      # Calculate variances.
      stds = torch.zeros_like(trainset[0][0])
      for ins, outs in trainset:
        stds += (ins - means)**2
      stds /= len(trainset)
      stds = stds**0.5
      
      # Save for future use.
      with open(path_means, "wb") as fout:
        torch.save(means, fout)
      with open(path_stds, "wb") as fout:
        torch.save(stds, fout)
    
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean = means, std = stds)
    ])
    
    trainset = f(root = path, train = True, download = True, transform = transform)
    testset = f(root = path, train = False, download = True, transform = transform)
    logging.info("Done.")
    return trainset, testset
  
  return res


#### DATASET LOADERS ###################################################

from torchvision import datasets


normed_MNIST = normed(datasets.MNIST)
normed_CIFAR10 = normed(datasets.CIFAR10)
normed_CIFAR100 = normed(datasets.CIFAR100)
