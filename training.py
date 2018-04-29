import torch
from torch.autograd import Variable

from lib import cuda

import logging


def validate(model, val_data, metrics):
  """Calculates <metrics> for <model> on given <val_data>."""
  if cuda:
    model.cuda()
  else:
    model.cpu()
  
  meter = {f.__name__ : 0.0 for f in metrics}
  count = 0
  
  for ins, tgts in val_data:
    if cuda:
      ins = ins.cuda()
      tgts = tgts.cuda()
    batch_size = ins.size()[0]
    count += batch_size
    
    outs = model(ins)
    for f in metrics:
      meter[f.__name__] += torch.sum(f(outs, tgts), dim = 0).item()
  
  for f in metrics:
    meter[f.__name__] /= count
  return meter


class Trainer:
  """Contains variables relevant for training, and facilitates their
  convenient reuse. Also contains training logic."""
  
  def __init__(
    self,
    train_data, val_data,
    criterion, metrics,
    batch_size,
    optimizer, scheduler = None, opt_kwargs = {}, sch_kwargs = {},
    num_epochs = 3,
    early_stopping = 1023456789 # almost like infinity
  ):
    """Initializes the default values for parameters."""
    self.train_data = train_data
    self.val_data = val_data
    self.model = None
    self.criterion = criterion
    self.metrics = metrics
    self.batch_size = batch_size
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.opt_kwargs = opt_kwargs
    self.sch_kwargs = sch_kwargs
    self.num_epochs = num_epochs
    
    # Early stopping is based on the first metric in the <metrics> list.
    self.early_stopping = early_stopping
  
  def set(self, **kwargs):
    """Changes the values imbued in this Trainer."""
    for name, value in kwargs.items():
      if name in ["model", "criterion", "metrics", "batch_size", "num_epochs", "optimizer", "cuda", "scheduler", "early_stopping"]:
        self.__setattr__(name, value)
      elif name[:4] == "sch_":
        self.sch_kwargs[name[4:]] = value
      else:
        self.opt_kwargs[name] = value
    return self
  
  def train(self):
    """Trains the model, and returns the training and validation metrics plotted in time."""
    if cuda:
      self.model.cuda()
    else:
      self.model.cpu()
    
    train_loader = torch.utils.data.DataLoader(self.train_data, self.batch_size, True, pin_memory = cuda)
    val_loader = torch.utils.data.DataLoader(self.val_data, self.batch_size, False, pin_memory = cuda)
    
    tH = {f.__name__ : [] for f in self.metrics}
    vH = {f.__name__ : [] for f in self.metrics}
    best_val = None
    best_model_state = None
    patience = self.early_stopping
    
    opt = self.optimizer(self.model.parameters(), **self.opt_kwargs)
    if self.scheduler:
      sch = self.scheduler(opt, **self.sch_kwargs)
    else:
      sch = None
    
    stats = {f.__name__ : 0.0 for f in self.metrics}
    count = 0
    
    for epoch in range(self.num_epochs):
      logging.info("Epoch %d/%d", epoch + 1, self.num_epochs)
      if sch:
        sch.step()
      
      milestone = 0.0
      done = 0
      for ins, tgts in train_loader:
        if cuda:
          ins = ins.cuda()
          tgts = tgts.cuda()
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
          tH[f.__name__].append(torch.mean(f(outs.data, tgts.data), dim = 0).item())
          stats[f.__name__] += torch.sum(f(outs.data, tgts.data), dim = 0).item()
        count += batch_size
        done += batch_size
        progress = done / len(self.train_data)
        if progress - milestone >= 0.1 or progress == 1:
          milestone = progress
          for f in self.metrics:
            stats[f.__name__] /= count
          logging.info("\tProgress %d%%, metrics: %s", int(100*progress), stats)
          stats = {f.__name__ : 0.0 for f in self.metrics}
          count = 0
      
      meter = validate(self.model, val_loader, self.metrics)
      logging.info("Validation metrics after epoch %d/%d: %s", epoch + 1, self.num_epochs, meter)
      for f in self.metrics:
        vH[f.__name__].append(meter[f.__name__])
      
      # Early stopping.
      curr_val = meter[self.metrics[0].__name__]
      if best_val is None or curr_val < best_val:
        best_val = curr_val
        best_model_state = self.model.state_dict()
        patience = self.early_stopping
      else:
        patience -= 1
        if patience <= 0:
          logging.info("Patience run out, stopping early. Best validation error was %.9f", best_val)
          break
    
    # At the end of it all, load the best model (validation_error-wise).
    self.model.load_state_dict(best_model_state)
    
    return {"train": tH, "val": vH}
