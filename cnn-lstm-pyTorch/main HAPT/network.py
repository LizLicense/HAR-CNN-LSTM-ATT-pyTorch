import torch
import torch.nn as nn


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64,
                      kernel_size=6, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(0.1)

        self.lstm = nn.LSTM(input_size=47, hidden_size=32, num_layers=1)
        self.tanh = torch.nn.Tanh()

        self.fc = nn.Linear(in_features=128*32, out_features=12)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # print("1", x.dtype, x.shape)
        x = x.permute(0, 2, 1)
        out = self.conv1(x)
        # print("c1", out.dtype, out.shape)
        out = self.conv2(out)
        # print("c2", out.dtype, out.shape)
        out = self.dropout(out)
        # print("dropout", out.shape, out.dtype)
        out, hidden = self.lstm(out)
        out = self.tanh(out)
        # print("lstm", out.dtype, out.shape)

        out = self.flatten(out)
        # print("flatten", out.dtype, out.shape)
        out = self.fc(out)
        # print("fc", out.dtype, out.shape)
        out = self.softmax(out)
        # print("sm", out.dtype, out.shape)

        return out