import logging
logging.basicConfig(format = "%(asctime)s %(message)s", datefmt = "%H:%M:%S", level = logging.INFO)

import torch
cuda = torch.cuda.is_available()
