import torch.nn as nn

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input, hidden=5):
        super().__init__()
        self.lstm = nn.LSTM(input, hidden, 1, True)
        self.linear = nn.Linear(hidden, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        out = self.linear(x)

        return out