
import torch
import torch.nn as nn

# prediction
class classifier(nn.Module): 
    def __init__(self):
        super(classifier, self).__init__()
        # print(hparams)
        self.logits = nn.Linear(in_features=64*64, out_features=6)

    def forward(self, x):
        # print(x.shape)
        x_flat = x.reshape(x.shape[0], -1)
        predictions = self.logits(x_flat)
        return predictions


##########################################################################################

class cnnNetwork_UCI(nn.Module):  
    def __init__(self):
        super(cnnNetwork_UCI, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels = 9, out_channels = 64, kernel_size=6, stride=1, padding=2),           
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size=3, stride=1, padding =2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(0.1)

        self.lstm = nn.LSTM(input_size=32, hidden_size=128, num_layers=1)
        self.tanh = torch.nn.Tanh()
        # self.attn = TemporalAttn(hidden_size=64)
        # self.fc = nn.Linear(in_features=64, out_features=6)
        self.fc = nn.Linear(in_features=128*128, out_features=6)
        self.softmax = torch.nn.Softmax(dim = 1)


    def forward(self, x):
        #print("1", x.dtype, x.shape)
        # x = x.permute(0,2,1) 
        out = self.conv1(x)
        #print("c1", out.dtype, out.shape)
        out = self.conv2(out)
        #print("c2", out.dtype, out.shape)
        # out = self.dropout(out)
        #print("dropout", out.shape, out.dtype)
        # out, hidden = self.lstm(out)
        # out = self.tanh(out)
        #print("lstm", out.dtype, out.shape)
        #attention
        # out, weights=self.attn(out)
        # out = self.flatten(out)
        #print("flatten", out.dtype, out.shape)
        # out = self.fc(out)
        #print("fc", out.dtype, out.shape)
        # out = self.softmax(out)
        #print("sm", out.dtype, out.shape)

        return out

# can try LSTM
class cnn1d_temporal(nn.Module):
    def __init__(self):
        super(cnn1d_temporal, self).__init__()

    def forward(self, x):
        return x

class cnnNetwork_HHAR(nn.Module):
    
    def __init__(self):
        super(cnnNetwork_HHAR, self).__init__()
        self.conv1 = nn.Sequential(

            nn.Conv1d(in_channels = 3, out_channels = 64, kernel_size=6, stride=1, padding=2),            
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels =64, out_channels = 128, kernel_size=3, stride=1, padding =2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(0.1)

        self.lstm = nn.LSTM(input_size=32, hidden_size=128, num_layers=1)
        self.tanh = torch.nn.Tanh()
        # self.attn = TemporalAttn(hidden_size=32)
        self.fc = nn.Linear(in_features=128*128, out_features=6)
        self.softmax = torch.nn.Softmax(dim = 1)


    def forward(self, x):
        # print("1", x.dtype, x.shape)
        out = self.conv1(x)
        # print("c1", out.dtype, out.shape)
        out = self.conv2(out)
        # print("c2", out.dtype, out.shape)
        # out = self.dropout(out)
        # print("dropout", out.shape, out.dtype)
        # out, hidden = self.lstm(out)
        # out = self.tanh(out)
        # print("lstm", out.dtype, out.shape)
        # out = self.flatten(out)
        # print("flatten", out.dtype, out.shape)
        # out = self.fc(out)
        # print("fc", out.dtype, out.shape)
        # out = self.softmax(out)
        # print("sm", out.dtype, out.shape)

        return out


class cnnNetwork_HAPT(nn.Module):
    
    def __init__(self):
        super(cnnNetwork_HAPT, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels = 3, out_channels = 64, kernel_size=6, stride=1, padding=2),            
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size=3, stride=1, padding =2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(0.1)

        self.lstm = nn.LSTM(input_size=47, hidden_size=128, num_layers=1)
        self.tanh = torch.nn.Tanh()

        self.fc = nn.Linear(in_features=128*128, out_features=12)
        self.softmax = torch.nn.Softmax(dim = 1)


    def forward(self, x):
        # print("1", x.dtype, x.shape)
        x = x.permute(0,2,1)
        out = self.conv1(x)
        # print("c1", out.dtype, out.shape)
        out = self.conv2(out)
        # print("c2", out.dtype, out.shape)
        # out = self.dropout(out)
        # print("dropout", out.shape, out.dtype)
        # out, hidden = self.lstm(out)
        # out = self.tanh(out)
        # print("lstm", out.dtype, out.shape)
        # out = self.flatten(out)
        # print("flatten", out.dtype, out.shape)
        # out = self.fc(out)
        # print("fc", out.dtype, out.shape)
        # out = self.softmax(out)
        # print("sm", out.dtype, out.shape)

        return out

    