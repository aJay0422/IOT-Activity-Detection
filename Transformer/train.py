import torch
import torch.nn as nn

from utils import get_loss_acc, prepare_data
from model import transformer_base, transformer_huge, transformer_large


def train(model, epochs, trainloader, testloader, optimizer, criterion, save_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Trained on {}".format(device))
    model.to(device)

    # train model
    best_test_acc = 0
    model.train()
    for epoch in range(epochs):
        for X_batch, Y_batch in trainloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # forward
            logits = model(X_batch)
            loss = criterion(logits, Y_batch)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate
        with torch.no_grad():
            model.eval()
            # evaluate train
            train_loss, train_acc = get_loss_acc(model, trainloader, criterion)
            # evaluate test
            test_loss, test_acc = get_loss_acc(model, testloader, criterion)

        print("Epoch{}/{}  train loss: {}  test loss: {}  train acc: {}  testacc: {}".format(epoch+1, epochs,
                                                                                             train_loss, test_loss,
                                                                                             train_acc, test_acc))
        # save model weights if  it's the best
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print("Saved")


if __name__ == "__main__":
    seed = 20220728
    trainloader, testloader = prepare_data(test_ratio=0.15, seed=seed)

    # train model
    model = transformer_huge()
    epochs = 500
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    train(model, epochs, trainloader, testloader, optimizer, criterion, "Transformer_huge.pth")
