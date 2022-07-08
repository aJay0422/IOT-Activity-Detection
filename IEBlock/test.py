import torch
import torch.nn as nn
from model import IEAutoEncoder, MLPBaseline
from utils import prepare_data, prepare_encoded_data, trainMLP


if __name__ == "__main__":
    # prepare data
    trainloader, testloader = prepare_data(test_ratio=0.15)

    # load models
    mlp_base = MLPBaseline()
    mlp_checkpoint = torch.load("MLPBaseline.pt")
    mlp_base.load_state_dict(mlp_checkpoint["model_state_dict"])

    ae = IEAutoEncoder(34, 100, 15, 25, 10, 15)
    ae_checkpoint = torch.load("Best_AE.pt")
    ae.load_state_dict(ae_checkpoint["model_state_dict"])

    # extract feature using autoencoder
    encoder = ae.encoder
    trainloader_encoded, testloader_encoded= prepare_encoded_data(encoder, trainloader, testloader)

    # train a new mlp
    mlp = MLPBaseline(1500, 600, 5)
    EPOCHS = 200
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    trained_mlp = trainMLP(mlp, EPOCHS, trainloader_encoded, testloader_encoded, optimizer, criterion, "MLP")

