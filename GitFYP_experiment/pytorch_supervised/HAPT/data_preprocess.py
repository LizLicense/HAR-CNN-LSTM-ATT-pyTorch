import numpy as np
from torch.utils.data import Dataset, DataLoader

# 561 features, 1 file, make it as 3 *187
# This is for parsing the X data, you can ignore it if you do not need preprocessing
def format_data_x(datafile):
    x_data = None
    item_data = np.loadtxt(datafile, dtype=np.float)
    if x_data is None:
        x_data = np.zeros((len(item_data), 1))
    x_data = np.hstack((x_data, item_data))
    x_data = x_data[:, 1:]

    X = None
    for i in range(len(x_data)):
        row = np.asarray(x_data[i, :])
        row = row.reshape(3, 187).T
        if X is None:
            # X = np.zeros((len(x_data), 128, 1))
            X = np.zeros((len(x_data), 187, 3))
        X[i] = row
    print(X.shape)
    return X


# This is for parsing the Y data, you can ignore it if you do not need preprocessing
def format_data_y(datafile):
    data = np.loadtxt(datafile, dtype=np.int) - 1
    YY = np.eye(12)[data]
    return YY


def onehot_to_label(y_onehot):
    a = np.argwhere(y_onehot == 1)
    return a[:, -1]


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


def normalize(x):
    x_min = x.min(axis=(0, 2, 3), keepdims=True)
    x_max = x.max(axis=(0, 2, 3), keepdims=True)
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm


# Load data function, if there exists parsed data file, then use it
# If not, parse the original dataset from scratch
def load_data(data_folder):
    import os
    if os.path.isfile(data_folder + 'data_har.npz') == True:
        data = np.load(data_folder + 'data_har.npz')
        X_train = data['X_train']
        Y_train = data['Y_train']
        X_test = data['X_test']
        Y_test = data['Y_test']
    else:
        # This for processing the dataset from scratch
        # After downloading the dataset, put it to somewhere that str_folder can find
        str_folder = data_folder + 'HAPT Data Set/'    #'/Users/lizliao/Downloads/GitFYP_experiment/Dataset/HAPT Data Set'

        str_train_files = str_folder + 'Train/X_train.txt' #train data
        str_train_y = str_folder + 'Train/y_train.txt' #train label
        
        str_test_files = str_folder + 'Test/X_test.txt' #test data
        str_test_y = str_folder + 'Test/y_test.txt' #test label


        X_train = format_data_x(str_train_files)
        X_test = format_data_x(str_test_files)
        Y_train = format_data_y(str_train_y)
        Y_test = format_data_y(str_test_y)

    return X_train, onehot_to_label(Y_train), X_test, onehot_to_label(Y_test)


def load(data_folder, batch_size=64):
    x_train, y_train, x_test, y_test = load_data(data_folder)
    # x_train, x_test = x_train.reshape(
    #     (-1, 9, 1, 128)), x_test.reshape((-1, 9, 1, 128))
    transform = None
    train_set = data_loader(x_train, y_train, transform)
    test_set = data_loader(x_test, y_test, transform)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# load('../Dataset/')


