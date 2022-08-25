import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class data_loader(Dataset):
    def __init__(self, samples, labels, t):
        self.samples = samples
        self.labels = labels
        self.T = t

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        if self.T:
            return self.T(sample), target
        else:
            return sample, target

    def __len__(self):
        return len(self.samples)


# file = "/Users/lizliao/Downloads/GitFYP_experiment/Dataset/HHAR Processed Data/HHAR_w/train.pt"
def load_data(data_folder):
    
    # accelerometer gyroscope a&g
    train_data = torch.load(data_folder + 'HHAR Processed Data/HHAR_w/accelerometer/train.pt')
    test_data = torch.load(data_folder + 'HHAR Processed Data/HHAR_w/accelerometer/test.pt')
    X_train = train_data['samples']
    Y_train = train_data['labels']
    X_test = test_data['samples']
    Y_test = test_data['labels']

    return X_train,Y_train,X_test,Y_test

def load(data_folder, batch_size):
    print(data_folder)
    x_train, y_train, x_test, y_test = load_data(data_folder)
    transform = None
    train_set = data_loader(x_train, y_train, transform)
    test_set = data_loader(x_test, y_test, transform)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader



# data_folder = '/Users/lizliao/Downloads/GitFYP_experiment/Dataset/'
# load(data_folder, 64)


