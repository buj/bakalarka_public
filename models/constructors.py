from .creation import *


#### Basic stuff #######################################################

base = {
  "start": identity,
  "conv": conv,
  "dense": dense,
  "dropout": identity   # No dropout by default.
}

def drop(layers):
  """Returns a set of layers, without dropout."""
  return {**layers, "dropout": dropout}


def io(layers):
  """Uses the IO-dense layer instead of the dense layer. Not recommended."""
  return {**layers, "dense": io_dense}


############ Activations ###############################################

def sgnlogd(layers):
  """Signed logarithm activation."""
  return {**layers,
    "conv": activated(layers["conv"], sgnlog),
    "dense": activated(layers["dense"], sgnlog)
  }


def relud(layers):
  """Relu activation."""
  relu_gain = nn.init.calculate_gain("relu")
  return {**layers,
    "conv": activated(layers["conv"], relu, relu_gain),
    "dense": activated(layers["dense"], relu, relu_gain)
  }


################ Normalizations ########################################

def bn(layers):
  """Batch norm before every activation."""
  return {**layers,
    "conv": batch_normed(layers["conv"]),
    "dense": batch_normed(layers["dense"])
  }


def ln(layers):
  """Layer normalization."""
  return {**layers,
    "conv": layer_normed(layers["conv"]),
    "dense": layer_normed(layers["dense"])
  }


################ Elementwise/channelwise shifting and scaling ##########

def sh(layers):
  """Elementwise/channelwise shift (add some value)."""
  return {**layers,
    "conv": channel_shifted(layers["conv"]),
    "dense": element_shifted(layers["dense"])
  }


def sc(layers):
  """Elementwise/channelwise scale (multiply by some value)."""
  return {**layers,
    "conv": channel_scaled(layers["conv"]),
    "dense": element_scaled(layers["dense"])
  }


################ Layerwise shifting and scaling ########################

def lsh(layers):
  """Layerwisse shift."""
  return {**layers,
    "conv": layer_shifted(layers["conv"]),
    "dense": layer_shifted(layers["dense"])
  }


def lsc(layers):
  """Layerwisse scale."""
  return {**layers,
    "conv": layer_scaled(layers["conv"]),
    "dense": layer_scaled(layers["dense"])
  }
