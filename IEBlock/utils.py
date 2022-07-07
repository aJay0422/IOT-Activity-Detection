import torch
from torch.utils.data import Dataset


class mydataset(Dataset):
    def __init__(self, X, Y):
        self.Data = torch.Tensor(X)
        self.Label = torch.LongTensor(Y)

    def __getitem__(self, item):
        return self.Data[item], self.Label[item]

    def __len__(self):
        return len(self.Label)