import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from augmentations import apply_transformation
import os
import utils as utils
# import configs 
datafolder ='UCI HAR Dataset/data/'
# configs = configs.Config()

#  oversample metod
def get_balance_class_oversample(x, y):
    return x, y


    
#class Load_Dataset(Dataset):
class data_loader(Dataset):
    def __init__(self, dataset, normalize, training_mode, augmentation, oversample=False):
        # self.samples = samples
        # self.labels = labels
        self.training_mode = training_mode
        self.augmentation = augmentation
        self.num_transformations = len(self.augmentation.split("_"))

        X_train = dataset["samples"]
        y_train = dataset["labels"]

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
        if self.training_mode == "self_supervised":
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
            print("supervised")
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
def load_data(data_folder, train_mode, augmentation, oversample):
    
        print("111")
        # load .pt file
        train_data = torch.load(data_folder + 'train.pt')
        val_data = torch.load(data_folder + 'val.pt')
        test_data = torch.load(data_folder + 'test.pt')
        
        # Loading datasets
        train_dataset = data_loader(train_data, True, train_mode, augmentation, oversample)
        val_dataset = data_loader(val_data, True, train_mode, augmentation, oversample)

        # Dataloaders
        # batch_size = hparams["batch_size"]
        batch_size = 64
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                shuffle=True, drop_last=True, num_workers=0)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                                shuffle=False, drop_last=True, num_workers=0)
        return train_loader, val_loader
        # X_train = train_data['samples']
        # Y_train = train_data['labels']
        # X_test = test_data['samples']
        # Y_test = test_data['labels']


# def load_data(data_folder, train_mode, ssl_method, augmentation, oversample):

# data_folder = '../datapt/'
# load_data(data_folder, "self_supervised","permute_timeShift_scale_noise", False)
