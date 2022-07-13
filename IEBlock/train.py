import torch
import torch.nn as nn
from utils import trainAE, trainMLP, prepare_data, get_loss_acc
from model import IEAutoEncoder, MLPBaseline


if __name__ == "__main__":
    trainloader, testloader = prepare_data(test_ratio=0.15)

    # train MLP baseline model
    model = MLPBaseline()
    EPOCHS = 300
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    trained_mlp = trainMLP(model, EPOCHS, trainloader, testloader, optimizer, criterion, model_name="MLPBaseline")

    _, mlp_base_train_acc = get_loss_acc(model, trainloader, nn.CrossEntropyLoss())
    _, mlp_base_test_acc = get_loss_acc(model, testloader, nn.CrossEntropyLoss())
    print("-" * 50)
    print("MLP Baseline model: train acc {}  test acc {}".format(mlp_base_train_acc, mlp_base_test_acc))

    # train autoencoder
    model = IEAutoEncoder(34, 100, 15, 25, 10, 15)
    EPOCHS = 1000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    trained_model = trainAE(model, EPOCHS, trainloader, testloader, optimizer, criterion)
