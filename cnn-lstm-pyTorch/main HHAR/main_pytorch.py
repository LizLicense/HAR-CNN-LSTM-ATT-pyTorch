from itertools import accumulate
import data_preprocess
import matplotlib.pyplot as plt
import network as net
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
result = []

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nepoch', type=int, default=10)
    parser.add_argument('--batchsize', type=int, default=32) #128 64
    parser.add_argument('--lr', type=float, default=.001) #0.0003
    parser.add_argument('--momentum', type=float, default=.9)
    parser.add_argument('--data_folder', type=str, default='Dataset/HHAR_w/')
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()
    return args


def train(model, optimizer, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()

    for e in range(args.nepoch):
        model.train()
        correct, total_loss = 0, 0
        total = 0
        for sample, label in train_loader:
            sample, label = sample.to(
                DEVICE).float(), label.to(DEVICE).long()
            output = model(sample)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum()
            
        acc_train = float(correct) * 100.0 / len(train_loader.dataset)

        # Testing
        acc_test = valid(model, test_loader)
        print(f'Epoch: [{e}/{args.nepoch}], loss:{total_loss / len(train_loader):.4f}, train_acc: {acc_train:.2f}, test_acc: {float(correct) * 100 / total:.2f}')
        result.append([acc_train, acc_test])
        result_np = np.array(result, dtype=float)
        np.savetxt('result_cnn-lstm_HHAR.csv', result_np, fmt='%.2f', delimiter=',')


def valid(model, test_loader):

    model.eval()
    
    # classes = ('Biking','Sitting', 'Standing', 'Walking', 'Stair Up' , 'Stair down')
    classes = ('stand', 'sit', 'walk', 'stairsup', 'stairsdown', 'bike')
    y_pred = []
    y_true = []
    f1 = []

    with torch.no_grad():
        correct, total = 0, 0
        for sample, label in test_loader:
            sample, label = sample.to(
                DEVICE).float(), label.to(DEVICE).long()
            output = model(sample)
            _, predicted = torch.max(output.data, 1)
            
            y_pred.extend(predicted)
            y_true.extend(label)
            f1.extend(f1)
            
            total += label.size(0)
            correct += (predicted == label).sum()
    acc_test = float(correct) * 100 / total

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('confusion matrix_HHAR.png')    
    f1 = f1_score(y_true, y_pred, average=None)# F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)
    return acc_test


def plot():
    data = np.loadtxt('result_cnn-lstm_HHAR.csv', delimiter=',')
    plt.figure()
    plt.plot(range(1, len(data[:, 0]) + 1),
             data[:, 0], color='blue', label='train')
    plt.plot(range(1, len(data[:, 1]) + 1),
             data[:, 1], color='red', label='test')
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Train and Test Accuracy', fontsize=16)
    plt.savefig('plot_cnn-lstm_HHAR.png')


if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)
    train_loader, test_loader = data_preprocess.load(
        args.data_folder, batch_size=args.batchsize)
    model = net.Network().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum)
    train(model, optimizer, train_loader, test_loader)
    result = np.array(result, dtype=float)
    np.savetxt('result_cnn-lstm_HHAR.csv', result, fmt='%.2f', delimiter=',')
    plot()
    
    