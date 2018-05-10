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
  
  model.eval()
  
  for ins, tgts in val_data:
    if cuda:
      ins = ins.cuda()
      tgts = tgts.cuda()
    batch_size = ins.size()[0]
    count += batch_size
    
    outs = model(ins)
    for f in metrics:
      meter[f.__name__] += torch.sum(f(outs, tgts), dim = 0).item()
  
  model.train()
  
  for f in metrics:
    meter[f.__name__] /= count
  return meter


class Trainer:
  """Contains variables relevant for training, and facilitates their
  convenient reuse. Also contains training logic."""
  
  def __init__(
    self, model,
    train_data, val_data,
    criterion, metrics,
    optimizer, scheduler = None, opt_kwargs = {"momentum": 0.9}, sch_kwargs = {},
    batch_size = 32,
    early_stopping = 1023456789 # almost like infinity
  ):
    """Initializes the default values for parameters."""
    self.hypers = locals()
    del self.hypers["self"]
    self.started = False
  
  @property
  def model(self):
    """Shortcut."""
    return self.hypers["model"]
  
  @property
  def metrics(self):
    """Shortcut."""
    return self.hypers["metrics"]
  
  def set(self, **kwargs):
    """Changes the values imbued in this Trainer. Only some values can be changed"""
    assert not self.started, "Training has already started, cannot change hyperparameters."
    for name, value in kwargs.items():
      if name in self.hypers:
        self.hypers[name] = value
      elif name[:4] == "sch_":
        self.hypers["sch_kwargs"][name[4:]] = value
      else:
        self.hypers["opt_kwargs"][name] = value
    return self
  
  def start(self):
    """Initialize stuff for the training."""
    assert not self.started, "Cannot start: training has already started"
    if cuda:
      self.model.cuda()
    else:
      self.model.cpu()
    
    self.train_loader = torch.utils.data.DataLoader(self.hypers["train_data"], self.hypers["batch_size"], True, pin_memory = cuda)
    self.val_loader = torch.utils.data.DataLoader(self.hypers["val_data"], self.hypers["batch_size"], False, pin_memory = cuda)
    
    self.tH = {f.__name__ : [] for f in self.metrics}
    self.vH = {f.__name__ : [] for f in self.metrics}
    self.best_val = None
    self.best_model_state = None
    self.patience = self.hypers["early_stopping"]
    
    self.opt = self.hypers["optimizer"](self.model.parameters(), **self.hypers["opt_kwargs"])
    if self.hypers["scheduler"]:
      self.sch = self.hypers["scheduler"](self.opt, **self.hypers["sch_kwargs"])
    else:
      self.sch = None
    
    self.epoch = 0
    self.started = True
  
  def step(self):
    """One epoch of training."""
    if not self.started:
      self.start()
    
    self.epoch += 1
    logging.info("Epoch %d", self.epoch)
    if self.sch:
      self.sch.step()
    
    milestone = 0.0
    count = 0
    done = 0
    stats = {f.__name__ : 0.0 for f in self.metrics}
    
    for ins, tgts in self.train_loader:
      if cuda:
        ins = ins.cuda()
        tgts = tgts.cuda()
      ins = Variable(ins)
      tgts = Variable(tgts)
      outs = self.model(ins)
      loss = self.hypers["criterion"](outs, tgts)
      self.opt.zero_grad()
      loss.backward()
      self.opt.step()
      
      # Log statistics.
      batch_size = ins.size()[0]
      for f in self.metrics:
        self.tH[f.__name__].append(torch.mean(f(outs.data, tgts.data), dim = 0).item())
        stats[f.__name__] += torch.sum(f(outs.data, tgts.data), dim = 0).item()
      count += batch_size
      done += batch_size
      progress = done / len(self.hypers["train_data"])
      if progress - milestone >= 0.1 or progress == 1:
        milestone = progress
        for f in self.metrics:
          stats[f.__name__] /= count
        logging.info("\tProgress %d%%, metrics: %s", int(100*progress), stats)
        stats = {f.__name__ : 0.0 for f in self.metrics}
        count = 0
    
    meter = validate(self.model, self.val_loader, self.metrics)
    logging.info("Validation metrics after epoch %d: %s", self.epoch, meter)
    for f in self.metrics:
      self.vH[f.__name__].append(meter[f.__name__])
    
    # Early stopping.
    curr_val = meter[self.metrics[0].__name__]
    if self.best_val is None or curr_val < self.best_val:
      self.best_val = curr_val
      self.best_model_state = self.model.state_dict()
      self.patience = self.hypers["early_stopping"]
    else:
      self.patience -= 1
      if self.patience <= 0:
        logging.info("Patience run out, stopping early. Best validation error was %.9f", self.best_val)
        return True
  
  def data(self):
    """Returns the training and validation statistics, and the best model."""
    assert self.started, "No data, since training has not started yet."
    return {"train": self.tH, "val": self.vH}
  
  def train(self, num_epochs):
    """Trains the model for <num_epochs> at most, maybe ledd, depending
    on the setting of early stopping."""
    for i in range(num_epochs):
      if self.step():
        break
