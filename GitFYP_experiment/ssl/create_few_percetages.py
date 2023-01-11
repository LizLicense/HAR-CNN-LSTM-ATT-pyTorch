import os
import torch
from sklearn.model_selection import train_test_split

data_dir = r"../uci_data"
output_dir = "../output_data"
os.makedirs(output_dir, exist_ok=True)

few_lbl_percentages = [1, 5, 10, 50, 75]


for percentage in few_lbl_percentages:
    data = torch.load(os.path.join(data_dir, f"train_100per.pt"))

    x_data = data["samples"].numpy()
    y_data = data["labels"].numpy()

    X_train, X_val, y_train, y_val = train_test_split(x_data, y_data, test_size=percentage/100, random_state=0)

    few_shot_dataset = {"samples": X_val, "labels": y_val}

    # saving data
    data_save = dict()
    data_save["samples"] = torch.from_numpy(X_val)
    data_save["labels"] = torch.from_numpy(y_val)
    torch.save(data_save, os.path.join(output_dir, f"train_{percentage}per.pt"))