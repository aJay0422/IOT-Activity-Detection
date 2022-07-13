import torch
import torch.nn as nn
import pandas as pd
from train import INITIAL_SEED
from model import IEAutoEncoder, MLPBaseline
from utils import prepare_data, prepare_encoded_data, trainMLP, get_loss_acc, get_mean_std


if __name__ == "__main__":
    summary = dict()
    summary["MLP Baseline Train Acc"] = []
    summary["MLP Baseline Test Acc"] = []
    summary["MLP reduced Train Acc"] = []
    summary["MLP reduced Test Acc"] = []

    for i in range(5):   # repeat 5 times
        seed = INITIAL_SEED + i
        # prepare data
        trainloader, testloader = prepare_data(test_ratio=0.15, seed=seed)

        # load models
        mlp_base = MLPBaseline()
        model_path = "MLPBaseline_checkpoints/MLPBaseline{}.pt".format(i+1)
        mlp_base_checkpoint = torch.load(model_path)
        mlp_base.load_state_dict(mlp_base_checkpoint["model_state_dict"])

        ae = IEAutoEncoder(34, 100, 15, 25, 10, 15)
        model_path = "Best_AE_checkpoints/Best_AE{}.pt".format(i+1)
        ae_checkpoint = torch.load(model_path)
        ae.load_state_dict(ae_checkpoint["model_state_dict"])

        # extract feature using autoencoder
        encoder = ae.encoder
        trainloader_encoded, testloader_encoded= prepare_encoded_data(encoder, trainloader, testloader)
        # X_all = torch.cat([trainloader_encoded.dataset.Data, testloader_encoded.dataset.Data], dim=0).detach().numpy()
        # Y_all = torch.cat([trainloader_encoded.dataset.Label, testloader_encoded.dataset.Label], dim=0).detach().numpy()
        # np.savez("all_feature_971_1500.npz", X=X_all, Y=Y_all)


        # train a new mlp
        model_name = "MLP_checkpoints/MLP{}".format(i+1)
        mlp = MLPBaseline(1500, 600, 5)
        EPOCHS = 300
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-5)
        criterion = nn.CrossEntropyLoss()
        trained_mlp = trainMLP(mlp, EPOCHS, trainloader_encoded, testloader_encoded, optimizer, criterion, model_name)

        # summary
        mlp = MLPBaseline(1500, 600, 5)
        mlp_checkpoint = torch.load("MLP.pt")
        mlp.load_state_dict(mlp_checkpoint["model_state_dict"])

        _, mlp_base_train_acc = get_loss_acc(mlp_base, trainloader, nn.CrossEntropyLoss())
        _, mlp_base_test_acc = get_loss_acc(mlp_base, testloader, nn.CrossEntropyLoss())
        _, mlp_train_acc = get_loss_acc(mlp, trainloader_encoded, nn.CrossEntropyLoss())
        _, mlp_test_acc = get_loss_acc(mlp, testloader_encoded, nn.CrossEntropyLoss())

        summary["MLP Baseline Train Acc"].append(mlp_base_train_acc)
        summary["MLP Baseline Test Acc"].append(mlp_base_test_acc)
        summary["MLP reduced Train Acc"].append(mlp_train_acc)
        summary["MLP reduced Test Acc"].append(mlp_test_acc)

        print("MLP Baseline model: train acc {}  test acc {}".format(mlp_base_train_acc, mlp_base_test_acc))
        print("MLP reduced  model: train acc {}  test acc {}".format(mlp_train_acc, mlp_test_acc))

    df = pd.DataFrame(columns=["Baseline Train Acc",
                               "Baseline Test Acc",
                               "Reduced Train Acc",
                               "Reduced Test Acc"])
    df.iloc[0, 0] = get_mean_std(summary["MLP Baseline Train Acc"])
    df.iloc[0, 1] = get_mean_std(summary["MLP Baseline Test Acc"])
    df.iloc[0, 2] = get_mean_std(summary["MLP reduced Train Acc"])
    df.iloc[0, 3] = get_mean_std(summary["MLP reduced Test Acc"])

    df.to_csv("summary_MLP_reduced.csv")