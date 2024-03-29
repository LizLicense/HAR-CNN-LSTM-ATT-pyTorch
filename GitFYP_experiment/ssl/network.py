
import torch
import torch.nn as nn

# prediction
class classifier(nn.Module): 
    def __init__(self, num, f):
        super(classifier, self).__init__()
        # print(hparams)
        self.logits = nn.Linear(in_features=f, out_features=num)

    def forward(self, x):
        # print(x.shape)
        x_flat = x.reshape(x.shape[0], -1)
        predictions = self.logits(x_flat)
        return predictions

# prediction
class ssl_classifier(nn.Module): 
    def __init__(self, num, f):
        super(ssl_classifier, self).__init__()
        # print(hparams)
        self.logits = nn.Linear(in_features=f, out_features=num)#64*64

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
            nn.Conv1d(in_channels = 9, out_channels = 64, kernel_size=6, stride=1, padding=2),   #64       
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size=3, stride=1, padding =2), #64 128
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
        nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size=3, stride=1, padding =2), 
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2)
    )

        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(0.1)

        self.fc = nn.Linear(in_features=128*128, out_features=6)
        self.softmax = torch.nn.Softmax(dim = 1)


    def forward(self, x):
        # x = x.permute(0,2,1) 
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.dropout(out)



        return out


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
        self.conv3 = nn.Sequential(
        nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size=3, stride=1, padding =2), 
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2)
    )
        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(0.1)

        self.fc = nn.Linear(in_features=128*128, out_features=6)
        self.softmax = torch.nn.Softmax(dim = 1)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.dropout(out)
       
        return out


class cnnNetwork_HAPT(nn.Module):
    
    def __init__(self):
        super(cnnNetwork_HAPT, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=3 , out_channels=32, kernel_size=6, stride=1, padding=2),            
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
        nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size=3, stride=1, padding =2), 
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2)
    )

        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(0.1)
        
        self.fc = nn.Linear(in_features=128*128, out_features=12)
        self.softmax = torch.nn.Softmax(dim = 1)


    def forward(self, x):
        x = x.permute(0,1,2)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.dropout(out)

        return out

    