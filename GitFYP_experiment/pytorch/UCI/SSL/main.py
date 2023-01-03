from itertools import accumulate
from data_loader import load_data
import matplotlib.pyplot as plt
from network import get_network_class
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import torch.nn.functional as F
from utils import AverageMeter, to_device, _save_metrics, copy_Files, starting_logs, save_checkpoint, _calc_metrics

DEVICE = torch.device('cpu')
# cpu = torch.device('cpu')
# DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


data_folder = '../datapt/'
save_dict = '../checkpoint_load/'
save_path = '../checkpoint_saved/'
result = []
f1_result = []
classes = ['WALKING', 'WALKING_UPSTAIRS',  'WALKING_DOWNSTAIRS','SITTING', 'STANDING', 'LAYING']

result_path='/Users/lizliao/Downloads/GitFYP_experiment/wip:tut:CNN-LSTM-ATT-torch/result/UCI/'
testAcc_csv=result_path+'result_cnn-lstm_UCI.csv'
f1_csv=result_path+'result_f1_cnn-lstm_UCI.csv'
confusion_img=result_path+'confusion matrix_cnn-lstm_UCI.png'
plot_img=result_path+'plot_cnn-lstm_UCI.png'

criterion = nn.CrossEntropyLoss()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nepoch', type=int, default=2) #50
    parser.add_argument('--batchsize', type=int, default=64) #128 64
    parser.add_argument('--lr', type=float, default=0.001) #0.0003
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--data_folder', type=str, default='../datapt/')
    # parser.add_argument('--data_folder', type=str, default=data_folder)
    parser.add_argument('--seed', type=int, default=10)

    parser.add_argument('--dataset', type=str, default='UCI', help='HAPT OR HHAR')
    parser.add_argument('--training_mode', type=str, default='ssl', 
                    help='Modes of choice: supervised, ssl, ft')
    parser.add_argument('--augmentation', type=str, default='noise_permute',    
                    help='permute_timeShift_scale_noise')
    parser.add_argument('--device', type=str, default='cpu',              
                    help='cpu or cuda or mps')
    parser.add_argument('--oversample', type=bool, default=False, 
                    help='apply oversampling or not?')

    args = parser.parse_args()
    return args

'''
# Supervised_original:
criterion = nn.CrossEntropyLoss()
# start training
loop nepoch:
    model.train()
    sample and label pass to device
    get the output
    get loss: criterion(output, label)
    loss = criterion(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    prediction: _, predicted = torch.max(output.data, 1) #_is the value, predicted is the value index
    count total: total += label.size(0)
    count correct prediction: correct += (predicted == label).sum()

# training completed
accuracy f1 rate:

do testing: valid(model, test_loader)
save result

----------------------------------------------------

SSL(stage 1)/FT(stage 2)/Supervised:
# start training
loop nepoch:
    model.train()
    for each step:
        sample data pass to device
        losses, model = algorithm.update(data) 
            if ssl: training with auxiliary labels, return loss
            if surpervised: training with true labels, return loss
    if not SSL: 
        do testing evaluate() # evaluate in each training
        calculate f1, acc
        save best model
    if SSL:
        save checkpoint
training completion


'''
def train(network, optimizer, train_loader, test_loader, training_mode, backbone_fe, backbone_temporal):
    if training_mode == "ft":
        # load saved models
        chekpoint = torch.load(os.path.join(data_folder,'checkpoint.pt'))
        # model=chekpoint

    #training
    for e in range(args.nepoch):
        # model.train()
        correct, total_loss = 0, 0
        best_f1 = 0
        best_acc = 0
        for sample in train_loader:
            # print(type(sample))
            #send data to device
            sample=to_device(sample, args.device)
            # sample = sample.to(DEVICE, dtype=torch.float32).float()

            if training_mode == "ssl":
                # data pass to update(), return model
                losses, model = ssl_update(backbone_fe, backbone_temporal, optimizer, classifier, sample)
                # cal metrics           
            elif training_mode != "ssl" : # supervised training or fine tuining 
                # to revise
                # losses, model = surpervised_update(sample)
                losses = surpervised_update(sample)
                # cal metrics f1 acc rate
                calc_results_per_run()
                # total_loss = total_loss + losses
                # testing
                valid(model, test_loader)
                # save best model 
                #model saved
                if f1 > best_f1:  # save best model based on best f1.
                    best_f1 = f1
                    best_acc = acc
                    cp_file = os.path.join(args.dataset, "_best_checkpoint.pt")
                    torch.save(save_path, cp_file)
                    save_results()
                    
            total_loss = total_loss + losses
        print(f'Epoch: [{e}/{args.nepoch}], loss:{total_loss / len(train_loader):.4f}')
        # print(f'Epoch: [{e}/{args.nepoch}], loss:{total_loss:.4f}')
    # save checkpoint
    print(training_mode)
    if training_mode == "ssl":       
        torch.save(model, "test.pt")
        # save_checkpoint(save_path,model, args.dataset)

def valid(model, test_loader, training_mode):

    model.eval()
    
    y_pred = []
    y_true = []
    f1 = []

    with torch.no_grad():
        correct, total = 0, 0
        
        for sample, label in test_loader:
            sample, label = sample.to(DEVICE, dtype=torch.float32).float(), label.to(DEVICE).long()
            
            if training_mode == "ssl":
                pass
            else:
                output = model(sample)
            _, predicted = torch.max(output.data, 1)
            
            y_pred.extend(predicted)
            y_true.extend(label)
            f1.extend(f1)
            
            total += label.size(0)
            correct += (predicted == label).sum()
    acc_test = float(correct) * 100 / total
    
    # save metrics
    _save_metrics(y_pred, y_true, save_path)
    # to revise
    # Build confusion matrix
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

def ssl_update(backbone_fe, backbone_temporal, optimizer,classifier, samples):
    # self.feature_extractor = backbone_fe
    # self.temporal_encoder = backbone_temporal
    # ====== Data =====================
    data = samples["transformed_samples"].float()
    labels = samples["aux_labels"].long()
    
    optimizer.zero_grad()

    features = backbone_fe(data)
    print("f1",features)
    features = features.flatten(1, 2)
    print("f2",features )
    logits = classifier(features)

    # Cross-Entropy loss
    loss = criterion(logits, labels)
    
    loss.backward()
    optimizer.step()
    
    # return loss.item()
    return {'Total_loss': loss.item()}, \
            [backbone_fe, backbone_temporal, classifier]

# to revise
def surpervised_update(samples):
    '''
        # ====== Data =====================
        data = samples['sample_ori'].float()
        labels = samples['class_labels'].long()

        # ====== Source =====================
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay= args.weight_decay)
        optimizer.zero_grad()

        # Src original features
        # features = feature_extractor(data)
        # features = temporal_encoder(features)
        # logits = classifier(features)
        features = model(data)
        # Cross-Entropy loss
        x_ent_loss = criterion(features, labels)

        x_ent_loss.backward()
        optimizer.step()

        return x_ent_loss.item()
        # return {'Total_loss': x_ent_loss.item()}, \
        #        [feature_extractor, temporal_encoder, classifier]
'''
def calc_results_per_run(self):
        self.acc, self.f1 = _calc_metrics(self.pred_labels, self.true_labels, self.dataset_configs.class_names)

def save_results(self):
        run_metrics = {'accuracy': self.best_acc, 'f1_score': self.best_f1}
        df = pd.DataFrame(columns=["acc", "f1"])
        df.loc[0] = [self.acc, self.f1]

        for (key, val) in run_metrics.items(): self.metrics[key].append(val)

        scores_save_path = os.path.join(self.home_path, self.scenario_log_dir, "scores.xlsx")
        df.to_excel(scores_save_path, index=False)
        self.results_df = df

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
# load_data(data_folder, "ssl","permute_timeShift_scale_noise", False)
if __name__ == '__main__':
    args = get_args()
    print(args.training_mode)
    torch.manual_seed(args.seed)
    # train_loader, test_loader = data_preprocess.load(
    #     args.data_folder, batch_size=args.batchsize, training_mode=args.training_mode)
    train_loader, test_loader = load_data(args.data_folder, args.training_mode, 
                                augmentation=args.augmentation, oversample=args.oversample)
    #load model
    # model = net.Network().to(DEVICE)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay= args.weight_decay)

    num_clsTran_tasks = len(args.augmentation.split("_"))
    # get Network
    backbone_fe = get_network_class(f"cnnNetwork")
    print(backbone_fe)
    backbone_temporal = get_network_class(f"cnn1d_temporal")
    print(backbone_temporal)
    classifier = get_network_class("classifier")
    print(classifier)

    # in_features 
    classifier = nn.Linear(in_features=128*128, out_features=num_clsTran_tasks)
    network = nn.Sequential(backbone_fe, backbone_temporal, classifier)
    # network = network.to(DEVICE)
    optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay= args.weight_decay, betas=(0.9, 0.99))


    # Train model
    train(network, optimizer, train_loader, test_loader, args.training_mode, backbone_fe, backbone_temporal)

    # result = np.array(result, dtype=float)
    # np.savetxt(testAcc_csv, result, fmt='%.2f', delimiter=',')
    # plot()
    # avg_F1_Acc()
    
    