from .creation import *


#### Defaults ##########################################################

default_layers = {
  "start": identity,
  "conv": activated(conv, relu, nn.init.calculate_gain("relu")),
  "dense": activated(dense, relu, nn.init.calculate_gain("relu")),
  "dropout": identity     # No dropout for now.
}


############ Io ########################################################

"""Have additional weights for each input and output of a dense layer,
that contribute to each adjacent weight."""
io_ = {**default_layers,
  "dense": activated(io_dense, relu, nn.init.calculate_gain("relu"))
}


############ Sgnlog ####################################################

"""Signed logarithm activation."""
sgnlog_ = {**default_layers,
  "conv": activated(conv, sgnlog),
  "dense": activated(dense, sgnlog)
}

"""bn_sgnlog_0.0007"""
bn_sgnlog_ = {**default_layers,
  "conv": activated(batch_normed(conv), sgnlog),
  "dense": activated(batch_normed(dense), sgnlog)
}


############ ReLU ######################################################

relu_gain = nn.init.calculate_gain("relu")


################ Batch norm variants ###################################

"""Batch norm before every activation."""
bn_relu_ = {**default_layers,
  "conv": activated(batch_normed(conv), relu, relu_gain),
  "dense": activated(batch_normed(dense), relu, relu_gain)
}


################ Layer normalization ###################################

"""Layer normalization, before each activation."""
ln_relu_ = {**default_layers,
  "conv": activated(layer_normed(conv), relu, relu_gain),
  "dense": activated(layer_normed(dense), relu, relu_gain)
}


################ Elementwise/channelwise shifting and scaling ##########

"""Scale prior to activation, then shift and scale. (There is an
implicit shifting before activation, done by the dense/conv layer.)"""
sc_relu_sh_sc_ = {**default_layers,
  "conv": channel_scaled(channel_shifted(activated(channel_scaled(conv), relu, relu_gain))),
  "dense": element_scaled(element_shifted(activated(element_scaled(dense), relu, relu_gain)))
}

"""Activate, then shift and scale."""
relu_sh_sc_ = {**default_layers,
  "conv": channel_scaled(channel_shifted(activated(conv, relu, relu_gain))),
  "dense": element_scaled(element_shifted(activated(dense, relu, relu_gain)))
}

"""Only scale. After activation."""
relu_sc_ = {**default_layers,
  "conv": channel_scaled(activated(conv, relu, relu_gain)),
  "dense": element_scaled(activated(dense, relu, relu_gain))
}

"""Only shift. After activation."""
relu_sh_ = {**default_layers,
  "conv": channel_shifted(activated(conv, relu, relu_gain)),
  "dense": element_shifted(activated(dense, relu, relu_gain))
}


################ Layerwise shifting and scaling ########################

"""Scale prior to activation, then shift and scale. All this is done
per layer."""
lsc_relu_limp_ = {**default_layers,
  "conv": layer_scaled(layer_shifted(activated(layer_scaled(conv), relu, relu_gain))),
  "dense": layer_scaled(layer_shifted(activated(layer_scaled(dense), relu, relu_gain)))
}


"""Activate, then shift and scale."""
relu_limp_ = {**default_layers,
  "conv": layer_scaled(layer_shifted(activated(conv, relu, relu_gain))),
  "dense": layer_scaled(layer_shifted(activated(dense, relu, relu_gain)))
}

"""Only scale. After activation."""
relu_lsc_ = {**default_layers,
  "conv": layer_scaled(activated(conv, relu, relu_gain)),
  "dense": layer_scaled(activated(dense, relu, relu_gain))
}

"""Only shift. After activation."""
relu_lsh_ = {**default_layers,
  "conv": layer_shifted(activated(conv, relu, relu_gain)),
  "dense": layer_shifted(activated(dense, relu, relu_gain))
}


################ Shifting and scaling combo ############################

"""Apply many shifts and scales."""
relu_combo = {**default_layers,
  "conv": layer_scaled(channel_scaled(layer_shifted(channel_shifted(activated(conv, relu, relu_gain))))),
  "dense": layer_scaled(element_scaled(layer_shifted(element_shifted(activated(dense, relu, relu_gain)))))
}
