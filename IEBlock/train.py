import torch
import torch.nn as nn
from utils import trainAE, trainMLP, prepare_data, get_loss_acc
from model import IEAutoEncoder, MLPBaseline


INITIAL_SEED = 20220712
if __name__ == "__main__":
    for i in range(5):   # repeat the whole experiment 5 times
        seed = INITIAL_SEED + i
        trainloader, testloader = prepare_data(test_ratio=0.15, seed=seed)

        # train MLP baseline model
        model_name = "MLPBaseline_checkpoints/MLPBaseline{}".format(i+1)
        model = MLPBaseline()
        EPOCHS = 300
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        criterion = nn.CrossEntropyLoss()
        trained_mlp = trainMLP(model, EPOCHS, trainloader, testloader, optimizer, criterion, model_name=model_name)

        _, mlp_base_train_acc = get_loss_acc(model, trainloader, nn.CrossEntropyLoss())
        _, mlp_base_test_acc = get_loss_acc(model, testloader, nn.CrossEntropyLoss())
        print("-" * 50)
        print("MLP Baseline model{}: train acc {}  test acc {}".format(i+1, mlp_base_train_acc, mlp_base_test_acc))

        # train autoencoder
        model_name = "Best_AE_checkpoints/Best_AE{}".format(i+1)
        model = IEAutoEncoder(34, 100, 15, 25, 10, 15)
        EPOCHS = 1000
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        trained_model = trainAE(model, EPOCHS, trainloader, testloader, optimizer, criterion, model_name)
