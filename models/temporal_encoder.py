import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=2,
            input_size=1024,
            hidden_size=1024,
            out_size=128,
            add_linear=True,
            bidirectional=False
    ):
        super(TemporalEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )

        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size*2, out_size)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, out_size)
        self.out_size = out_size

    def forward(self, x):
        n, t, f = x.shape
        x = x.permute(1, 0, 2)  # NTF -> TNF
        self.gru.flatten_parameters()
        y, _ = self.gru(x)
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(t, n, self.out_size)
        y = y.permute(1, 0, 2)  # TNF -> NTF
        return y
