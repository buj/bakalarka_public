from .creation import *


#### Meta stuff ########################################################

def act_wrap(f):
  """Should <f> be applied to 'before' or 'after'? With this decorator,
  we can determine that based on context (whether the activation is still
  identity or not)."""
  def res(layers):
    res = {**layers}
    when = "before"
    if layers["act"] is identity:
      when = "after"
    res[when] = f(layers[when])
    return res
  res.__name__ = f.__name__
  return res


#### Basic stuff #######################################################

base = {
  "conv": xavier_init(1, conv()),
  "dense": xavier_init(1, dense()),
  "before": identity,
  "act": identity,
  "after": identity,
  "dropout": identity
}


def biased(k, layers):
  """Returns a set of layers where dense and conv's biases are learning
  <k> times faster."""
  return {**layers,
    "dense": bias_lr(k, layers["dense"]),
    "conv": bias_lr(k, layers["conv"])
  }


def drop(layers):
  """Returns a set of layers with dropout."""
  return {**layers, "dropout": dropout}


def io(layers):
  """Uses the IO-dense layer instead of the dense layer. Not recommended."""
  return {**layers, "dense": io_dense}


############ Activations ###############################################

def sgnlog(layers):
  """Signed logarithm activation."""
  return {**layers,
    "act": sgnlog_act
  }


def relu(layers):
  """Relu activation."""
  relu_gain = nn.init.calculate_gain("relu")
  return {**layers,
    "conv": xavier_init(relu_gain, layers["conv"]),
    "dense": xavier_init(relu_gain, layers["dense"]),
    "act": relu_act
  }


################ Normalizations ########################################

bn = act_wrap(batch_norm)
ln = act_wrap(layer_norm)


################ Elementwise/channelwise shifting and scaling ##########

sh = act_wrap(shift)
sc = act_wrap(scale)
psc = act_wrap(pscale)

csh = act_wrap(channel_shift)
csc = act_wrap(channel_scale)
cpsc = act_wrap(channel_pscale)

np_sc = act_wrap(negpos_scale)


################ Layerwise shifting and scaling ########################

lsh = act_wrap(layer_shift)
lsc = act_wrap(layer_scale)
lpsc = act_wrap(layer_pscale)
