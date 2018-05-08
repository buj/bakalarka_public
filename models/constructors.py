from .creation import *


#### Defaults ##########################################################

default_layers = {
  "start": identity,
  "conv": activated(conv, relu, nn.init.calculate_gain("relu")),
  "pool": max_pool,     # Not actually used.
  "dense": activated(dense, relu, nn.init.calculate_gain("relu")),
  "dropout": identity   # No dropout for now.
}


############ Io ########################################################

"""Have additional weights for each input and output of a dense layer,
that contribute to each adjacent weight."""
io_layers = {**default_layers,
  "dense": activated(io_dense, relu, nn.init.calculate_gain("relu"))
}


############ Sgnlog ####################################################

"""Signed logarithm activation."""
sgnlog_layers = {**default_layers,
  "conv": activated(conv, sgnlog),
  "dense": activated(dense, sgnlog)
}

"""bn_sgnlog_0.0007"""
bn_sgnlog_layers = {**default_layers,
  "start": batch_normed(identity),
  "conv": activated(batch_normed(conv), sgnlog),
  "dense": activated(batch_normed(dense), sgnlog)
}


############ ReLU ######################################################

relu_gain = nn.init.calculate_gain("relu")


################ Batch norm variants ###################################

"""Normalize only input layer."""
inbn_relu_layers = {**default_layers,
  "start": batch_normed(identity)
}

"""Batch norm before every activation."""
bn_relu_layers = {**default_layers,
  "start": batch_normed(identity),
  "conv": activated(batch_normed(conv), relu, relu_gain),
  "dense": activated(batch_normed(dense), relu, relu_gain)
}


################ Layer norm ############################################

"""Layer normalization, before each activation."""
ln_relu_layers = {**default_layers,
  "start": batch_normed(identity),
  "conv": activated(batch_normed(conv), relu, relu_gain),
  "dense": activated(batch_normed(dense), relu, relu_gain)
}


################ Elementwise/channelwise shifting and scaling ##########

"""Scale prior to activation, then shift and scale. (There is an
implicit shifting before activation, done by the dense/conv layer.)"""
sc_relu_sh_sc_layers = {**default_layers,
  "start": element_scaled(element_shifted(identity)),
  "conv": channel_scaled(channel_shifted(activated(channel_scaled(conv), relu, relu_gain))),
  "dense": element_scaled(element_shifted(activated(element_scaled(dense), relu, relu_gain)))
}

"""Activate, then shift and scale."""
relu_sh_sc_layers = {**default_layers,
  "start": element_scaled(element_shifted(identity)),
  "conv": channel_scaled(channel_shifted(activated(conv, relu, relu_gain))),
  "dense": element_scaled(element_shifted(activated(dense, relu, relu_gain)))
}

"""Only scale. After activation."""
relu_sc_layers = {**default_layers,
  "start": element_scaled(identity),
  "conv": channel_scaled(activated(conv, relu, relu_gain)),
  "dense": element_scaled(activated(dense, relu, relu_gain))
}

"""Only shift. After activation."""
relu_sh_layers = {**default_layers,
  "start": element_shifted(identity),
  "conv": channel_shifted(activated(conv, relu, relu_gain)),
  "dense": element_shifted(activated(dense, relu, relu_gain))
}
