import torch
import torch.nn as nn
import math


class ElectricityLSTM(nn.Module):
    def __init__(self, seq_length, hidden_dim=128, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM( input_size=1, hidden_size=hidden_dim, batch_first=True)
        #self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.classifier1 = nn.Linear(hidden_dim, num_classes)
        self.classifier2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        x = x.unsqueeze(-1)

        _, (h_n, _) = self.lstm(x)

        out1 = self.classifier1(h_n.reshape(x.shape[0], -1))
        out2 = self.classifier2(h_n.reshape(x.shape[0], -1))

        #return self.hidden2label(h_n.reshape(x.shape[0], -1))
        return dict(
            logits_l=out1, #logits_drp
            logits_r=out2  #logits_spk
        )
    
    def private_params(self):
        return ['private_left.weight', 'private_left.bias', 'private_right.weight', 'private_right.bias']