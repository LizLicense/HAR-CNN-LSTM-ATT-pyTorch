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
import torch.nn.functional as F

DEVICE = torch.device('cpu')
# cpu = torch.device('cpu')
# DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

data_folder = '../datapt/'
result = []
f1_result = []
classes = ['WALKING', 'WALKING_UPSTAIRS',  'WALKING_DOWNSTAIRS','SITTING', 'STANDING', 'LAYING']

result_path='/Users/lizliao/Downloads/GitFYP_experiment/wip:tut:CNN-LSTM-ATT-torch/result/UCI/'
testAcc_csv=result_path+'result_cnn-lstm_UCI.csv'
f1_csv=result_path+'result_f1_cnn-lstm_UCI.csv'
confusion_img=result_path+'confusion matrix_cnn-lstm_UCI.png'
plot_img=result_path+'plot_cnn-lstm_UCI.png'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nepoch', type=int, default=50) #50
    parser.add_argument('--batchsize', type=int, default=64) #128 64
    parser.add_argument('--lr', type=float, default=.001) #0.0003
    parser.add_argument('--momentum', type=float, default=.9)
    parser.add_argument('--data_folder', type=str, default='../datapt/')
    # parser.add_argument('--data_folder', type=str, default=data_folder)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--training_mode', type=str, default='self_supervised', 
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
    args = parser.parse_args()
    return args


def train(model, optimizer, train_loader, test_loader, training_mode):
    criterion = nn.CrossEntropyLoss()
    # get model
    if training_mode == "self_supervised":
        print(training_mode)
        loss, model = ssl_update(sample)
    elif training_mode == "ft":
        print(training_mode)
        # load saved models
        chkpoint = torch.load('checkpoint.pt')

        loss, model = surpervised_update(sample)

    for e in range(args.nepoch):
        model.train()
        correct, total_loss = 0, 0
        total = 0
        best_f1 = 0
        best_acc = 0
        for sample, label in train_loader:
            #send to device
            sample, label = sample.to(DEVICE, dtype=torch.float32).float(), label.to(DEVICE).long()

            
            optimizer.zero_grad()

            if training_mode == "self_supervised":
                # save the best model
                # to revise
                save_checkpoint(self.home_path, model, self.dataset, self.dataset_configs, self.scenario_log_dir,
                            self.hparams)
                        
            elif training_mode != "self_supervised" : # supervised training or fine tuining 
                # re-organize the testing and evaluate part
                # to revise
                output = model(sample)
                print(training_mode)
                predictions, features = output
                loss = criterion(predictions, label)
                total_acc.append(label.eq(predictions.detach().argmax(dim=1)).float().mean())
                loss = criterion(output, label)
                loss.backward()
            
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1) #_is the value, predicted is the value index
            total += label.size(0)
            correct += (predicted == label).sum()
            
        acc_train = float(correct) * 100.0 / len(train_loader.dataset)
        
 
        # Testing
        if training_mode == "self_supervised":
            total_acc = 0
        else:
            acc_test = valid(model, test_loader)
            # print(f'Epoch: [{e}/{args.nepoch}], loss:{total_loss / len(train_loader):.4f}, train_acc: {acc_train:.2f}, test_acc: {float(correct) * 100 / total:.2f}')
        print(f'Epoch: [{e}/{args.nepoch}], loss:{total_loss / len(train_loader):.4f}, train_acc: {acc_train:.2f}, test_acc: {acc_test:.2f}')
        result.append([acc_train, acc_test])
        result_np = np.array(result, dtype=float)
        np.savetxt(testAcc_csv, result_np, fmt='%.2f', delimiter=',')

def valid(model, test_loader, training_mode):

    model.eval()
    
    y_pred = []
    y_true = []
    f1 = []

    with torch.no_grad():
        correct, total = 0, 0
        
        for sample, label in test_loader:
            sample, label = sample.to(DEVICE, dtype=torch.float32).float(), label.to(DEVICE).long()
            
            if training_mode == "self_supervised":
                pass
            else:
                output = model(sample)
            # _, predicted = torch.max(output.data, 1)
            _, predicted = torch.max(output.data, 1)
            
            y_pred.extend(predicted)
            y_true.extend(label)
            f1.extend(f1)
            
            total += label.size(0)
            correct += (predicted == label).sum()
    acc_test = float(correct) * 100 / total
    
    # to revise
    # Build confusion matrix -> as a sperate function
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(confusion_img)    
    f1 = f1_score(y_true, y_pred, average=None)# F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)
    f1_result.append(f1)
    f1_result_np = np.array(f1_result, dtype=float)
    np.savetxt(f1_csv, f1_result_np, fmt='%.4f', delimiter=',')
    print("f1: ", np.average(f1))
    return acc_test

def ssl_update(self, samples):
    # ====== Data =====================
    data = samples["transformed_samples"].float()
    labels = samples["aux_labels"].long()

    self.optimizer.zero_grad()

    features = self.feature_extractor(data)
    features = features.flatten(1, 2)
    
    logits = self.classifier(features)

    # Cross-Entropy loss
    loss = self.cross_entropy(logits, labels)

    loss.backward()
    self.optimizer.step()

    return {'Total_loss': loss.item()}, \
            [self.feature_extractor, self.temporal_encoder, self.classifier]

def surpervised_update(self, samples):
        # ====== Data =====================
        data = samples['sample_ori'].float()
        labels = samples['class_labels'].long()

        # ====== Source =====================
        self.optimizer.zero_grad()

        # Src original features
        features = self.feature_extractor(data)
        features = self.temporal_encoder(features)
        logits = self.classifier(features)

        # Cross-Entropy loss
        x_ent_loss = self.cross_entropy(logits, labels)

        x_ent_loss.backward()
        self.optimizer.step()

        return {'Total_loss': x_ent_loss.item()}, \
               [self.feature_extractor, self.temporal_encoder, self.classifier]
def plot():
    data = np.loadtxt(testAcc_csv, delimiter=',')
    plt.figure()
    plt.plot(range(1, len(data[:, 0]) + 1),
             data[:, 0], color='blue', label='train')
    plt.plot(range(1, len(data[:, 1]) + 1),
             data[:, 1], color='red', label='test')
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Train and Test Accuracy', fontsize=16)
    plt.savefig(plot_img)

def avg_F1_Acc():
    # # f1 avg
    data_f1 = np.loadtxt(f1_csv, delimiter=',')
    df_f1 = pd.DataFrame(data_f1, columns = classes)
    #save f1 avg to excel
    # mean_df=df_f1.mean().to_frame(name="average")
    # mean_trans=mean_df.transpose()
    # df_avg=pd.concat([df_f1,mean_trans])
    # df_avg.to_csv(f1_csv)
    total=0
    for i in df_f1.mean():
        total=total+i
    print(f"f1 average:{total/len(classes):.4f}")

    # # accuracy avg
    data_acc = np.loadtxt(testAcc_csv, delimiter=',')
    df_acc = pd.DataFrame(data_acc, columns = ["train_acc", "test_acc"])
    # save acc avg to excel
    mean_df=df_acc.mean().to_frame(name="average")
    mean_trans=mean_df.transpose()
    #save acc avg to excel
    # df_avg=pd.concat([df_acc,mean_trans])
    # df_avg.to_csv(testAcc_csv)
    trainAvg = mean_trans.loc["average", "train_acc"]
    testAvg = mean_trans.loc["average", "test_acc"]
    print(f"train acc avg: {trainAvg:.2f}, test acc avg: {testAvg:.2f}")

# data_folder = '../datapt/'
# load_data(data_folder, "self_supervised","permute_timeShift_scale_noise", False)
if __name__ == '__main__':
    args = get_args()
    print(args.training_mode)
    torch.manual_seed(args.seed)
    train_loader, test_loader = data_preprocess.load(
        args.data_folder, batch_size=args.batchsize, training_mode=args.training_mode)
    #load model
    model = net.Network().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    # Train model
    train(model, optimizer, train_loader, test_loader, args.training_mode)

    result = np.array(result, dtype=float)
    np.savetxt(testAcc_csv, result, fmt='%.2f', delimiter=',')
    plot()
    avg_F1_Acc()
    
    