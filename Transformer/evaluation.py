import torch
import numpy as np

from model import transformer_base, transformer_large, transformer_huge
from utils import get_loss_acc, count_parameters, prepare_data, draw_confusion_matrix


test_accs = []
seed = 20220728
for i in range(5):
    this_seed = seed + i
    trainloader, testloader = prepare_data(test_ratio=0.15, seed=this_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "experiment/Transformer_large_{}.pth".format(i + 1)
    model = transformer_large()
    model.load_state_dict(torch.load(model_path, map_location=device))

    loss, acc = get_loss_acc(model, testloader, torch.nn.CrossEntropyLoss())
    test_accs.append(acc)

print("Mean accuracy is {:.1f}%".format(np.mean(test_accs) * 100), end="  ")
print("with std {:.1f}%".format(np.std(test_accs) * 100))



