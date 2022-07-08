import torch
import torch.nn as nn
from utils import trainAE, trainMLP, prepare_data
from model import IEAutoEncoder, MLPBaseline


if __name__ == "__main__":
    trainloader, testloader = prepare_data(test_ratio=0.15)

    # train MLP baseline model
    model = MLPBaseline()
    EPOCHS = 200
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    trained_mlp = trainMLP(model, EPOCHS, trainloader, testloader, optimizer, criterion, model_name="MLPBaseline")

    # train autoencoder
    model = IEAutoEncoder(34, 100, 15, 25, 10, 15)
    EPOCHS = 500
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    trained_model = trainAE(model, EPOCHS, trainloader, testloader, optimizer, criterion)
