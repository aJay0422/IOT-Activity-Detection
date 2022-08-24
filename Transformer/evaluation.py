import torch
import numpy as np

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
net.load_state_dict(torch.load("experiment/Transformer_huge_1.pth", map_location=device))

trainloader, testloader = prepare_data(seed=20220728)
with torch.no_grad():
    net.eval()
    Y_pred = []
    Y_true = []
    for X_batch, Y_batch in testloader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        logits = net(X_batch)
        pred = torch.argmax(logits, dim=1)
        Y_pred.append(pred.cpu().numpy())
        Y_true.append(Y_batch.cpu().numpy())
    Y_pred = np.concatenate(Y_pred, axis=0)
    Y_true = np.concatenate(Y_true, axis=0)

draw_confusion_matrix(Y_pred, Y_true, save=True)
print(np.mean(Y_pred == Y_true))


