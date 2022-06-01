import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def basic_block(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.Sigmoid()
    )

class Autoencoder(nn.Module):
    def __init__(self, layers=[784, 2000, 1000, 500, 30]):
        super(Autoencoder, self).__init__()
        # 利用 basic_block 和 layers 构造 encoder 与 decoder.
        self.encoder = nn.Sequential(*[basic_block(i,j) for i,j in zip(layers[:-1],layers[1:])])
        self.decoder = nn.Sequential(*[basic_block(i,j) for i,j in zip(layers[:0:-1],layers[-2::-1])])

    def forward(self, x):
        hidden = self.encoder(x)
        reconstructed = self.decoder(hidden)
        return reconstructed
