from sklearn.model_selection import train_test_split
import torch
import os
import numpy as np

data_dir = 'HAPT Data Set/'
output_dir = 'output'

# subject_data = np.loadtxt(f'Train/subject_id_train.txt')
# Samples
train_data = np.loadtxt(f'Train/X_train.txt')
X_test = np.loadtxt(f'Test/X_test.txt')
print(train_data.shape, X_test.shape)
# labelss
train_labels = np.loadtxt(f'Train/y_train.txt')
train_labels -= np.min(train_labels)
y_test = np.loadtxt(f'Test/y_test.txt')
y_test -= np.min(y_test)
print(train_labels.shape, y_test.shape)


X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_train)
dat_dict["labels"] = torch.from_numpy(y_train)
torch.save(dat_dict, os.path.join(output_dir, "train.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_val)
dat_dict["labels"] = torch.from_numpy(y_val)
torch.save(dat_dict, os.path.join(output_dir, "val.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_test)
dat_dict["labels"] = torch.from_numpy(y_test)
torch.save(dat_dict, os.path.join(output_dir, "test.pt"))