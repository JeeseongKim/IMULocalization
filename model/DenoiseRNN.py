import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Denoise_RNN(nn.Module):
    def __init__(self, inp_sz, hidden_sz, num_layers, dropout, bidirectional):
        super(Denoise_RNN, self).__init__()

        self.LSTM = torch.nn.LSTM(input_size=inp_sz, hidden_size=2, num_layers=num_layers, bias=True, Batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, inp):

        LSTMoup = self.LSTM(inp)
        oup = LSTMoup.permute(1, 0, 2)

        return oup

