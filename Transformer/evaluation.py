import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from model import transformer_base, transformer_large, transformer_huge
from utils import get_loss_acc, count_parameters, prepare_data, draw_confusion_matrix, prepare_data_3D


# test_accs = []
# seed = 20220728
# for i in range(5):
#     this_seed = seed + i
#     trainloader, testloader = prepare_data(test_ratio=0.20, seed=this_seed)
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model_path = "experiment/Transformer_huge_{}.pth".format(i + 1)
#     model = transformer_huge()
#     model.load_state_dict(torch.load(model_path, map_location=device))
#
#     loss, acc = get_loss_acc(model, testloader, torch.nn.CrossEntropyLoss())
#     test_accs.append(acc)
#
# print("Mean accuracy is {:.1f}%".format(np.mean(test_accs) * 100), end="  ")
# print("with std {:.1f}%".format(np.std(test_accs) * 100))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = transformer_huge().to(device)


all_accs = {}
# no shuffle
for i in range(5):
    seed = 20220728 + i
    trainloader, testloader = prepare_data(seed=seed, shuffle_frame=False)
    all_accs["no shuffle"] = []
    net.load_state_dict(torch.load(f"experiment_shuffle/no_shuffle/Transformer_huge_no_shuffle_{i+1}.pth", map_location=device))
    _, acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())
    all_accs["no shuffle"].append(acc)

# shuffle
for shuffle_id in [1,2,3]:
    for i in range(5):
        seed = 20220728 + i
        trainloader, testloader = prepare_data(seed=seed, shuffle_frame=True)
        all_accs[f"shuffle {shuffle_id}"] = []
        net.load_state_dict(
            torch.load(f"experiment_shuffle/shuffle/Transformer_huge_shuffle{shuffle_id}_{i+1}.pth", map_location=device))
        _, acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())
        all_accs[f"shuffle {shuffle_id}"].append(acc)

for name in all_accs.keys():
    print(name, end=" ")
    print("Mean: {}".format(np.mean(all_accs[name])), end=" ")
    print("Std: {}".format(np.std(all_accs[name])))

