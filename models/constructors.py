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
  "conv": conv(1),
  "dense": dense(1),
  "before": identity,
  "act": identity,
  "after": identity,
  "dropout": identity
}

def drop(layers):
  """Returns a set of layers, without dropout."""
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
    "conv": conv(relu_gain),
    "dense": dense(relu_gain),
    "act": relu_act
  }


################ Normalizations ########################################

bn = act_wrap(batch_norm)
ln = act_wrap(layer_norm)


################ Elementwise/channelwise shifting and scaling ##########

sh = act_wrap(shift)
sc = act_wrap(scale)
psc = act_wrap(pscale)

np_sc = act_wrap(negpos_scale)


################ Layerwise shifting and scaling ########################

lsh = act_wrap(layer_shift)
lsc = act_wrap(layer_scale)
lpsc = act_wrap(layer_pscale)
