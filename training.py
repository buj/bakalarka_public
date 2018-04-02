import torch
from torch.autograd import Variable

import logging


def validate(model, val_data, metrics):
  """Calculates <metrics> for <model> on given <val_data>."""
  meter = {f.__name__ : 0.0 for f in metrics}
  count = 0
  
  for ins, tgts in val_data:
    batch_size = ins.size()[0]
    count += batch_size
    
    outs = model(ins)
    for f in metrics:
      meter[f.__name__] += torch.sum(f(outs, tgts), dim = 0)[0]
  
  for f in metrics:
    meter[f.__name__] /= count
  return meter


class Trainer:
  """Contains variables relevant for training, and facilitates their
  convenient reuse. Also contains training logic."""
  
  def __init__(self, train_data, val_data, criterion, metrics, batch_size, optimizer, opt_kwargs = {}, num_epochs = 3):
    """Initializes the default values for parameters."""
    self.train_data = train_data
    self.val_data = val_data
    self.model = None
    self.criterion = criterion
    self.metrics = metrics
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.optimizer = optimizer
    self.opt_kwargs = opt_kwargs
  
  def set(self, **kwargs):
    """Changes the values imbued in this Trainer."""
    for name, value in kwargs.items():
      if name in ["model", "criterion", "metrics", "batch_size", "num_epochs", "optimizer"]:
        self.__setattr__(name, value)
      else:
        self.opt_kwargs[name] = value
    return self
  
  def train(self):
    """Trains the model, and returns the training and validation metrics plotted in time."""
    train_loader = torch.utils.data.DataLoader(self.train_data, self.batch_size, True)
    val_loader = torch.utils.data.DataLoader(self.val_data, self.batch_size, False)
    tH = {f.__name__ : [] for f in self.metrics}
    vH = {f.__name__ : [] for f in self.metrics}
    opt = self.optimizer(self.model.parameters(), **self.opt_kwargs)
    
    stats = {f.__name__ : 0.0 for f in self.metrics}
    count = 0
    
    for epoch in range(self.num_epochs):
      logging.info("Epoch %d/%d", epoch + 1, self.num_epochs)
      
      milestone = 0.0
      done = 0
      for ins, tgts in train_loader:
        ins = Variable(ins)
        tgts = Variable(tgts)
        outs = self.model(ins)
        loss = self.criterion(outs, tgts)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # Log statistics.
        batch_size = ins.size()[0]
        for f in self.metrics:
          tH[f.__name__].append(torch.mean(f(outs.data, tgts.data), dim = 0)[0])
          stats[f.__name__] += torch.sum(f(outs.data, tgts.data), dim = 0)[0]
        count += batch_size
        done += batch_size
        progress = done / len(self.train_data)
        if progress - milestone >= 0.1 or progress == 1:
          milestone = progress
          for f in self.metrics:
            stats[f.__name__] /= count
          logging.info("Progress %d%%, metrics: %s", int(100*progress), stats)
          stats = {f.__name__ : 0.0 for f in self.metrics}
          count = 0
      
      meter = validate(self.model, val_loader, self.metrics)
      for f in self.metrics:
        vH[f.__name__].append(meter[f.__name__])
    
    return {"train": tH, "val": vH}
