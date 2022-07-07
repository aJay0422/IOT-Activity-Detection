import torch
import torch.nn as nn


class IEBlock(nn.Module):
    def __init__(self, H, W, F_p, F):
        super(IEBlock, self).__init__()
        self.mlp1 = nn.Linear(H, F_p)
        self.mlp2 = nn.Linear(W, W)
        self.mlp3 = nn.Linear(F_p, F)

    def forward(self, x):
        b, h, w = x.shape   # x: (b, h, w)
        x = x.transpose(-2, -1)   # x: (b, w, h)
        x = x.flatten(0, 1)   # x: (bw, h)
        x = self.mlp1(x)   # x: (bw, F_p)
        x = x.reshape(b, w, -1)   # x: (b, w, F_p)
        x = x.transpose(-2, -1)   # x: (b, F_p, w)
        x = self.mlp2(x)   # x: (b, F_p, w)
        x = x.transpose(-2, -1)   # x: (b, w, F_p)
        x = self.mlp3(x)   # x: (b, w, F)
        x = x.transpose(-2, -1)   # x: (b, F, w)

        return x

class IEAutoEncoder(nn.Module):
    def __init__(self, H, W, F_p1, F1, F_p2, F2):
        super(IEAutoEncoder, self).__init__()
        # encoder
        self.encoder1 = IEBlock(H, W, F_p1, F1)
        self.encoder2 = IEBlock(F1, W, F_p2, F2)
        self.encoder = nn.Sequential(*[
            self.encoder1,
            self.encoder2
        ])
        # decoder
        self.decoder1 = IEBlock(F2, W, F_p2, F1)
        self.decoder2 = IEBlock(F1, W, F_p1, H)
        self.decoder = nn.Sequential(*[
            self.decoder1, self.decoder2
        ])

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x