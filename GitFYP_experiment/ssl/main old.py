from itertools import accumulate
import collections
from data_loader import load_data
import matplotlib.pyplot as plt
import network as net
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

data_folder = '../datapt/'
save_dict = './checkpoint_load/'
save_path = './checkpoint_saved/'
best_cp = './checkpoint_best/'
result = []
f1_result = []
classes = ['WALKING', 'WALKING_UPSTAIRS',  'WALKING_DOWNSTAIRS','SITTING', 'STANDING', 'LAYING']
result_path='./result/'

testAcc_csv=result_path+'result_cnn-lstm_UCI.csv'
f1_csv=result_path+'result_f1_cnn-lstm_UCI.csv'
confusion_img=result_path+'confusion matrix_cnn-lstm_UCI.png'
plot_img=result_path+'plot_cnn-lstm_UCI.png'

criterion = nn.CrossEntropyLoss()


def get_args():
    parser = argparse.ArgumentParser()
    # ===================parameters===========================
    parser.add_argument('--nepoch', type=int, default=30) #50
    parser.add_argument('--batchsize', type=int, default=64) #128 64
    parser.add_argument('--lr', type=float, default=0.001) #0.0003
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--betas', type=float, default=(0.9, 0.999))
    parser.add_argument('--seed', type=int, default=10)
    # ===================settings===========================
    parser.add_argument('--data_folder', type=str, default='../datapt/')
    parser.add_argument('--dataset', type=str, default='UCI', help='HAPT OR HHAR')
    parser.add_argument('--training_mode', type=str, default='ft', 
                    help='Modes of choice: supervised, ssl, ft')
    parser.add_argument('--augmentation', type=str, default='permute_timeShift_scale_noise',    
                    help='negate_permute_timeShift_scale_noise')
    parser.add_argument('--device', type=str, default='mps',              
                    help='cpu or mps')
    parser.add_argument('--oversample', type=bool, default=False, 
                    help='apply oversampling or not?')

    args = parser.parse_args()
    return args

def train(train_loader, test_loader, training_mode):
    #logger
    logger = starting_logs(args.dataset, training_mode, result_path)

    # get Network - ssl/supervised
    backbone_fe = net.cnnNetwork()
    backbone_temporal = net.cnn1d_temporal()
    classifier = net.classifier()

    # Average meters
    loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

    if training_mode == "ft":
        # load saved models
        chekpoint = torch.load(os.path.join(save_dict,'UCI_SSL_checkpoint.pt'))
        # backbone_fe.load_state_dict(chekpoint["fe"])
        backbone_fe.load_state_dict(chekpoint["fe"])
        # print(backbone_fe.load_state_dict(chekpoint["fe"]))

    elif training_mode not in ["ssl", "supervised"]:
        print("Traiinng mode not found!")


    network = nn.Sequential(backbone_fe, backbone_temporal, classifier)
    optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay= args.weight_decay, betas=(0.9, 0.99))

    #training
    for e in range(args.nepoch):
        total_loss = 0, 0
        best_f1 = 0
        best_acc = 0
        for sample in train_loader:
            #send data to device
            sample=to_device(sample, args.device)

            if training_mode == "ssl":
                # data pass to update(), return model
                losses, model = ssl_update(backbone_fe, backbone_temporal, classifier, sample, optimizer)
                # losses, model = ssl_update(cnnNetwork, net.cnn1d_temporal(), classifier, sample, optimizer)
                # cal metrics           
                for key, val in losses.items():
                    loss_avg_meters[key].update(val, args.batchsize)
                
            
            elif training_mode != "ssl" : # supervised training or fine tuining 
                losses, model = surpervised_update(backbone_fe, backbone_temporal, classifier, sample, optimizer)
                # cal metrics f1 acc rate
                for key, val in losses.items():
                    loss_avg_meters[key].update(val, args.batchsize)
                
                
                # testing
                y_pred, y_true=valid(test_loader, net.cnnNetwork(), net.cnn1d_temporal(), classifier,)
                acc_test, f1 = calc_results_per_run(y_pred, y_true)
                
                # save best model 
                if f1 > best_f1:  # save best model based on best f1.
                    best_f1 = f1
                    best_acc = acc_test
                    save_checkpoint(best_cp, model, args.dataset)
                    save_results(best_acc,best_f1)
                    _save_metrics(y_pred, y_true, result_path, classes)
           
        # logging
        logger.debug(f'printf[Epoch : {e}/{args.nepoch}]')
        for key, val in loss_avg_meters.items():
            logger.debug(f'{key}\t: {val.avg:2.4f}')
            if training_mode != "ssl":
                logger.debug(f'Acc:{acc_test:2.4f} \t F1:{f1:2.4f} (best: {best_f1:2.4f})')
        logger.debug(f'-------------------------------------')


    # save checkpoint
    if training_mode == "ssl":       
        save_checkpoint(save_path, model, args.dataset)

def valid(test_loader, feature_extractor, temporal_encoder, classifier):

    feature_extractor.eval()
    temporal_encoder.eval()
    classifier.eval()

    total_loss_ = []
    y_pred = np.array([])
    y_true = np.array([])

    # send to cpu
    classifier = classifier.to("cpu")

    with torch.no_grad():
        for data in test_loader:
            data_samples = to_device(data, "cpu") # prediction in cpu

            data = data_samples['sample_ori'].float()
            labels = data_samples['class_labels'].long()
            
            # forward pass
            features = feature_extractor(data)
            features = temporal_encoder(features)
            predictions = classifier(features)

            # compute loss
            loss = F.cross_entropy(predictions, labels)
            total_loss_.append(loss.item())
            pred = predictions.detach().argmax(dim=1)  # get the index of the max log-probability

            y_pred = np.append(y_pred, pred.cpu().numpy().tolist())
            y_true = np.append(y_true, labels.data.cpu().numpy().tolist())

        # trg_loss = torch.tensor(total_loss_).mean()  # average loss
    
    return  y_pred, y_true
   

def ssl_update(backbone_fe, backbone_temporal,classifier, samples, optimizer):
    # self.feature_extractor = backbone_fe
    # self.temporal_encoder = backbone_temporal
    network = nn.Sequential(backbone_fe, backbone_temporal, classifier)
    network.to(args.device)
    # optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay= args.weight_decay, betas=(0.9, 0.99))
    
    # ====== Data =====================
    data = samples["transformed_samples"].float()
    labels = samples["aux_labels"].long()

    optimizer.zero_grad()
    features = backbone_fe(data)

    features = features.flatten(1,2)
    logits = classifier(features)

    # Cross-Entropy loss
    loss = criterion(logits, labels)
    
    loss.backward()
    optimizer.step()
    
    # return loss.item()
    return {'Total_loss': loss.item()}, \
            [backbone_fe, backbone_temporal, classifier]

# to revise
def surpervised_update(backbone_fe, backbone_temporal, classifier, samples, optimizer):
    # self.feature_extractor = backbone_fe
    # self.temporal_encoder = backbone_temporal
    
    # ====== Data =====================
    network = nn.Sequential(backbone_fe, backbone_temporal, classifier)
    network.to(args.device)
    # optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay= args.weight_decay, betas=args.betas)

    data = samples['sample_ori'].float()
    labels = samples['class_labels'].long()

    optimizer.zero_grad()
    features = backbone_fe(data)
    features = backbone_temporal(features)
    logits = classifier(features)

    # Cross-Entropy loss
    loss = criterion(logits, labels)
    
    loss.backward()
    optimizer.step()
    
    # return loss.item()
    return {'Total_loss': loss.item()}, \
            [backbone_fe, backbone_temporal, classifier]


def calc_results_per_run(pred_labels, true_labels):
        acc, f1 = _calc_metrics(pred_labels, true_labels, classes)
        return acc, f1

def save_results(best_acc, best_f1):
    metrics = {'accuracy': [], 'f1_score': []}
    run_metrics = {'accuracy': best_acc, 'f1_score': best_f1}
    df = pd.DataFrame(columns=["acc", "f1"])
    df.loc[0] = [best_acc, best_f1]

    for (key, val) in run_metrics.items(): metrics[key].append(val)

    scores_save_path = os.path.join(result_path, "scores.xlsx")
    df.to_excel(scores_save_path, index=False)
    # results_df = df

def create_folder(folder_name,train_mode):
    log_dir = os.path.join(".", train_mode, folder_name)
    os.makedirs(log_dir, exist_ok=True)

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

if __name__ == '__main__':
    args = get_args()
    print(args.training_mode)
    torch.manual_seed(args.seed)
    train_loader, test_loader = load_data(args.data_folder, args.training_mode, 
                                augmentation=args.augmentation, oversample=args.oversample)

    num_clsTran_tasks = len(args.augmentation.split("_"))

    # Train model
    train(train_loader, test_loader, args.training_mode)

    # result = np.array(result, dtype=float)
    # np.savetxt(testAcc_csv, result, fmt='%.2f', delimiter=',')
    # plot()
    # avg_F1_Acc()
    
    
