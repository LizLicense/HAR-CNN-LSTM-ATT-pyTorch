import torch
import random
import os
import sys
import logging
import numpy as np
import pandas as pd
from shutil import copy
from datetime import datetime

from sklearn.metrics import classification_report, accuracy_score


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def starting_logs(data_type, ssl_method, sleep_model, train_mode, exp_log_dir, fold_id):
    log_dir = os.path.join(exp_log_dir, "_fold_" + str(fold_id), train_mode)
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')
    logger.debug(f'Method:  {ssl_method}')
    logger.debug(f'Model:  {sleep_model}')
    logger.debug("=" * 45)
    logger.debug(f'Fold ID: {fold_id}')
    logger.debug("=" * 45)
    return logger, log_dir


def save_checkpoint(home_path, model, dataset):
    save_dict = {
        "dataset": dataset,
        # "configs": dataset_configs.__dict__,
        # "hparams": dict(hparams),
        "fe": model[0].state_dict(),
        "te": model[1].state_dict(),
        "clf": model[2].state_dict()
    }
    # save classification report
    save_path = os.path.join(home_path, dataset, "checkpoint.pt")

    torch.save(save_dict, save_path)


def _calc_metrics(pred_labels, true_labels, classes_names):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    r = classification_report(true_labels, pred_labels, target_names=classes_names, digits=6, output_dict=True)
    accuracy = accuracy_score(true_labels, pred_labels)

    return accuracy * 100, r["macro avg"]["f1-score"] * 100


def _save_metrics(pred_labels, true_labels, home_path):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # r = classification_report(true_labels, pred_labels, target_names=classes_names, digits=6, output_dict=True)
    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)

    df = pd.DataFrame(r)
    accuracy = accuracy_score(true_labels, pred_labels)
    df["accuracy"] = accuracy
    df = df * 100

    # save classification report
    file_name = "classification_report.xlsx"
    report_Save_path = os.path.join(home_path, file_name)
    df.to_excel(report_Save_path)


import collections


def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError("Input must contain tensor, dict or list, found {type(input)}")


def copy_Files(destination):
    destination_dir = os.path.join(destination, "MODEL_BACKUP_FILES")
    os.makedirs(destination_dir, exist_ok=True)
    copy("main.py", os.path.join(destination_dir, "main.py"))
    copy("dataloader/dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy("dataloader/ts_augmentations.py", os.path.join(destination_dir, "ts_augmentations.py"))
    copy(f"models/models.py", os.path.join(destination_dir, f"models.py"))
    copy(f"models/loss.py", os.path.join(destination_dir, f"loss.py"))
    copy("algorithms.py", os.path.join(destination_dir, "algorithms.py"))
    copy(f"configs/data_configs.py", os.path.join(destination_dir, f"data_configs.py"))
    copy(f"configs/hparams.py", os.path.join(destination_dir, f"hparams.py"))
    copy(f"trainer.py", os.path.join(destination_dir, f"trainer.py"))
    copy("utils.py", os.path.join(destination_dir, "utils.py"))



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


def normalize(x):
    x_min = x.min(axis=(0, 2, 3), keepdims=True)
    x_max = x.max(axis=(0, 2, 3), keepdims=True)
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm


def onehot_to_label(y_onehot):
    a = np.argwhere(y_onehot == 1)
    return a[:, -1]