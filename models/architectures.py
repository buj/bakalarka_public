import torch
from torch import nn

from lib.functional import flatten
from .general import Functional
from .creation import NetDescription


#### DESCRIPTION OF CONVNET1 ###########################################

names1 = [
  "start",
  "conv11", "conv12", "conv13",
  "conv21", "conv22", "conv23",
  "conv31", "conv32", "conv33",
  "flatten", "dense1", "dense2"
]
table1 = {
  **{"conv{}{}".format(i, j): nn.Conv2d for j in range(1, 4) for i in range(1, 4)},
  "flatten": Functional,
  "dense1": nn.Linear,
  "dense2": nn.Linear
}
args1 = {
  "conv11": [3, 6, 3],
  "conv12": [6, 12, 3],
  "conv13": [12, 12, 2],
  "conv21": [12, 24, 3],
  "conv22": [24, 48, 3],
  "conv23": [48, 48, 2],
  "conv31": [48, 96, 3],
  "conv32": [96, 192, 3],
  "conv33": [192, 192, 2],
  "flatten": [flatten],
  "dense1": [3072, 200],
  "dense2": [200, 10]
}
kwargs1 = {
  "conv11": {"padding": 1},
  "conv12": {"padding": 1},
  "conv13": {"stride": 2},
  "conv21": {"padding": 1},
  "conv22": {"padding": 1},
  "conv23": {"stride": 2},
  "conv31": {"padding": 1},
  "conv32": {"padding": 1},
  "conv33": {"stride": 2}
}
sizes1 = {
  "start": (3, 32, 32),
  "conv11": (6, 32, 32),
  "conv12": (12, 32, 32),
  "conv13": (12, 16, 16),
  "conv21": (24, 16, 16),
  "conv22": (48, 16, 16),
  "conv23": (48, 8, 8),
  "conv31": (96, 8, 8),
  "conv32": (192, 8, 8),
  "conv33": (192, 4, 4),
  "flatten": (3072,),
  "dense1": (200,),
  "dense2": (10,)
}

convnet1 = NetDescription("convnet1", names1, table1, args1, kwargs1, sizes1)


#### CONVNET2 ##########################################################

names2 = [
  "start",
  "conv11", "conv12", "conv13",
  "conv21", "conv22", "conv23",
  "conv31", "conv32", "conv33",
  "flatten", "dense"
]
table2 = {
  **{"conv{}{}".format(i, j): nn.Conv2d for j in range(1, 4) for i in range(1, 4)},
  "flatten": Functional,
  "dense": nn.Linear,
}
args2 = {
  "conv11": [3, 60, 3],
  "conv12": [60, 120, 3],
  "conv13": [120, 120, 2],
  "conv21": [120, 240, 3],
  "conv22": [240, 480, 3],
  "conv23": [480, 480, 2],
  "conv31": [480, 240, 3],
  "conv32": [240, 120, 3],
  "conv33": [120, 120, 2],
  "flatten": [flatten],
  "dense": [1920, 10],
}
kwargs2 = {
  "conv11": {"padding": 1},
  "conv12": {"padding": 1},
  "conv13": {"stride": 2},
  "conv21": {"padding": 1},
  "conv22": {"padding": 1},
  "conv23": {"stride": 2},
  "conv31": {"padding": 1},
  "conv32": {"padding": 1},
  "conv33": {"stride": 2}
}
sizes2 = {
  "start": (3, 32, 32),
  "conv11": (60, 32, 32),
  "conv12": (120, 32, 32),
  "conv13": (120, 16, 16),
  "conv21": (240, 16, 16),
  "conv22": (480, 16, 16),
  "conv23": (480, 8, 8),
  "conv31": (240, 8, 8),
  "conv32": (120, 8, 8),
  "conv33": (120, 4, 4),
  "flatten": (1920,),
  "dense": (10,)
}

convnet2 = NetDescription("convnet2", names2, table2, args2, kwargs2, sizes2)
