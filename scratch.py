import numpy as np
import torch.nn as nn
import torch

from Transformer.model import Transformer

net = Transformer()
X = torch.randn(55, 34, 100)
print(net(X).size())










stop = None