import numpy as np
import torch.nn as nn
import torch

from Transformer.model import Transformer


LABEL_DECODER = {0:"no_interaction",
                 1:"open_close_fridge",
                 2:"put_back_item",
                 3:"screen_interaction",
                 4:"take_out_item"}

all_data = np.load("feature_archive/" + "all_feature_interp951.npz")
Y_all = all_data["Y"]
for i in range(5):
    print(LABEL_DECODER[i], np.sum(Y_all == i))