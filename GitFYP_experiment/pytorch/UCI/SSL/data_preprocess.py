import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from augmentations import DataTransform
import os
datafolder ='UCI HAR Dataset/data/'
import configs 
configs = configs.Config()
# ref: https://github.com/emadeldeen24/TS-TCC.git
# This is for parsing the X data, you can ignore it if you do not need preprocessing
def format_data_x(datafile):
    x_data = None
    for item in datafile:
        item_data = np.loadtxt(item, dtype=np.float) #dtype=np.float
        # print(torch.tensor(1.0).dtype)
        if x_data is None:
            x_data = np.zeros((len(item_data), 1))
        x_data = np.hstack((x_data, item_data))
    x_data = x_data[:, 1:]
    print("1", x_data.shape) #(7352, 1152)
    X = None
    for i in range(len(x_data)):
        row = np.asarray(x_data[i, :])
        # row = row.reshape(9, 128).T
        row = row.reshape(9, 128).T
        if X is None:
            # X = np.zeros((len(x_data), 128, 9))
            X = np.zeros((len(x_data), 128, 9))
        X[i] = row
    print("1", X.shape) #(7352, 128, 3)
    return X


# This is for parsing the Y data, you can ignore it if you do not need preprocessing
def format_data_y(datafile):
    data = np.loadtxt(datafile, dtype=np.int) - 1
    YY = np.eye(6)[data]
    return YY


# Load data function, if there exists parsed data file, then use it
# If not, parse the original dataset from scratch
def load_data(data_folder):
    
    if os.path.isfile(data_folder + 'train.pt') == True:
        print("111")
        train_data = torch.load(data_folder + 'train.pt')
        valid_data = torch.load(data_folder + 'val.pt')
        test_data = torch.load(data_folder + 'test.pt')
        
        X_train = train_data['samples']
        Y_train = train_data['labels']
        X_test = test_data['samples']
        Y_test = test_data['labels']
    else:
        # This for processing the dataset from scratch
        # After downloading the dataset, put it to somewhere that str_folder can find
        str_folder = data_folder + datafolder
        INPUT_SIGNAL_TYPES = [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_"
        ]

        str_train_files = [str_folder + 'train/' + 'Inertial Signals/' + item + 'train.txt' for item in
                           INPUT_SIGNAL_TYPES]
        str_test_files = [str_folder + 'test/' + 'Inertial Signals/' +
                          item + 'test.txt' for item in INPUT_SIGNAL_TYPES]
        str_train_y = str_folder + 'train/y_train.txt'
        str_test_y = str_folder + 'test/y_test.txt'

        X_train = format_data_x(str_train_files)
        X_test = format_data_x(str_test_files)
        Y_train = format_data_y(str_train_y)
        Y_test = format_data_y(str_test_y)

        print(str_folder)
    return X_train, Y_train, X_test, Y_test
    # return X_train, onehot_to_label(Y_train), X_test, onehot_to_label(Y_test)




def onehot_to_label(y_onehot):
    a = np.argwhere(y_onehot == 1)
    return a[:, -1]


class data_loader(Dataset):
    
    def __init__(self, samples, labels, training_mode):
        self.samples = samples
        self.labels = labels
        self.training_mode = training_mode
        # self.T = t
        # add SSL

        if training_mode == "self_supervised":  # no need to apply Augmentations in other modes
            # self.aug1, self.aug2 = DataTransform(self.x_data)
            self.aug1, self.aug2 = DataTransform(self.samples, configs)

    #add SSL
    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        if self.training_mode == "self_supervised":
            return sample, target, self.aug1[index], self.aug2[index]
            # return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        # if self.T:
        #     return self.T(sample), target
        else:
            return sample, target

    def __len__(self):
        return len(self.samples)


def normalize(x):
    x_min = x.min(axis=(0, 2, 3), keepdims=True)
    x_max = x.max(axis=(0, 2, 3), keepdims=True)
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm


def load(data_folder, batch_size, training_mode):
    x_train, y_train, x_test, y_test = load_data(data_folder)
    # SSL
    transform = None
    train_set = data_loader(x_train, y_train, training_mode)
    # add valid?
    test_set = data_loader(x_test, y_test, training_mode)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader



data_folder = '../datapt/'
load(data_folder, 64, "self_supervised")
