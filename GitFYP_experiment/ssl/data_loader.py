import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from augmentations import apply_transformation
import os
import utils as utils

#  oversample metod
def get_balance_class_oversample(x, y):
    """
    from deepsleepnet https://github.com/akaraspt/deepsleepnet/blob/master/deepsleep/utils.py
    Balance the number of samples of all classes by (oversampling):
        1. Find the class that has the largest number of samples
        2. Randomly select samples in each class equal to that largest number
    """
    class_labels = np.unique(y)
    n_max_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_max_classes < n_samples:
            n_max_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        n_samples = len(idx)
        n_repeats = int(n_max_classes / n_samples)
        tmp_x = np.repeat(x[idx], n_repeats, axis=0)
        tmp_y = np.repeat(y[idx], n_repeats, axis=0)
        n_remains = n_max_classes - len(tmp_x)
        if n_remains > 0:
            sub_idx = np.random.permutation(idx)[:n_remains]
            tmp_x = np.vstack([tmp_x, x[sub_idx]])
            tmp_y = np.hstack([tmp_y, y[sub_idx]])
        balance_x.append(tmp_x)
        balance_y.append(tmp_y)
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)

    return balance_x, balance_y

    
#class Load_Dataset(Dataset):
class data_loader(Dataset):
    def __init__(self, dataset, normalize, training_mode, augmentation, oversample):
        # self.samples = samples
        # self.labels = labels
        self.training_mode = training_mode
        self.augmentation = augmentation
        self.num_transformations = len(self.augmentation.split("_"))

        X_train = dataset["samples"]
        y_train = dataset["labels"]
        print(X_train.shape)
        print(y_train.shape)
        if oversample and "ft" not in training_mode:  # if fine-tuning, it shouldn't be on oversampled data
            X_train, y_train = get_balance_class_oversample(X_train, y_train)
        
        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train)
            y_train = torch.from_numpy(y_train).long()
        
        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        self.x_data = X_train
        self.y_data = y_train

        self.num_channels = X_train.shape[1]

        if normalize:
            # Assume datashape: num_samples, num_channels, seq_length
            data_mean = torch.FloatTensor(self.num_channels).fill_(0).tolist()  # assume min= number of channels
            data_std = torch.FloatTensor(self.num_channels).fill_(1).tolist()  # assume min= number of channels
            data_transform = transforms.Normalize(mean=data_mean, std=data_std)
            self.transform = data_transform
        else:
            self.transform = None

        self.len = X_train.shape[0]

    def __getitem__(self, index):
        # sample, target = self.samples[index], self.labels[index]
        if self.training_mode == "self_supervised" or self.training_mode == "ssl":
            transformed_samples = apply_transformation(self.x_data[index], self.augmentation)
            order = np.random.randint(self.num_transformations)
            transformed_sample = transformed_samples[order]
            sample = {
                'transformed_samples': transformed_sample,
                'aux_labels': int(order),
                'sample_ori': self.x_data[index].squeeze(-1)
            }
            
        # else include ft and surpervised
        else:
            sample = {
                'sample_ori': self.x_data[index].squeeze(-1),
                'class_labels': int(self.y_data[index])
            }
        return sample

    def __len__(self):
        return self.len

# data generator
# if there exists parsed data file, then use it
# If not, parse the original dataset from scratch
def load_data(data_folder, train_mode, data_percentage, augmentation, oversample):
        # print("111")
        # load .pt file
        if train_mode == "ssl":  # we always want to do ssl with full labels.
            data_percentage = 100
        train_data = torch.load(os.path.join(data_folder, f'train_{data_percentage}per.pt'))
        val_data = torch.load(os.path.join(data_folder + 'val.pt'))
        test_data = torch.load(os.path.join(data_folder + 'test.pt'))
        
        # Loading datasets
        train_dataset = data_loader(train_data, True, train_mode, augmentation, oversample)
        val_dataset = data_loader(val_data, True, train_mode, augmentation, oversample)

        # Dataloaders
        batch_size = 64
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                shuffle=True, drop_last=True, num_workers=0)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                                shuffle=False, drop_last=True, num_workers=0)

        return train_loader, val_loader




# data_folder = '../datapt/'
# load_data(data_folder, "self_supervised","permute_timeShift_scale_noise", False)
