import collections
from data_loader import load_data
import network as net
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import seaborn as sn
import pandas as pd
import statistics
import torch.nn.functional as F
from utils import AverageMeter, to_device, _save_metrics, starting_logs, save_checkpoint, _calc_metrics, create_folder

# update with dataset
classes = {"UCI_classes" :["WALKING", "WALKING_UPSTAIRS",
           "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"],
        "HHAR_classes" : ["stand", "sit", "walk", "stairsup", "stairsdown", "bike"],
        "HAPT_classes":["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", 
                    "STANDING", "LAYING", "STAND_TO_SIT", "SIT_TO_STAND", "SIT_TO_LIE", 
                        "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND" ]}

#loss function
criterion = nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss()
mse_loss = nn.MSELoss()

f1_list=[]
acc_list=[]

def get_args():
    parser = argparse.ArgumentParser()
    # ===================parameters===========================
    parser.add_argument("--nepoch", type=int, default=50)  # 50
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)  # 0.0003
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--betas", type=float, default=(0.9, 0.999))
    parser.add_argument("--seed", type=int, default=10)
    # ===================settings===========================
    parser.add_argument("--data_percentage", type=str, help="1, 5, 10, 50, 75, 100") #, default="10"
    parser.add_argument("--training_mode", type=str, 
                        help="Modes of choice: supervised, ssl(self-supervised), ft(fine-tune)")# default="ssl",
    parser.add_argument("--dataset", type=str,  help="UCI or HAPT OR HHAR") # default="UCI",
    parser.add_argument("--classes", type=str,  help="UCI_classes or HAPT_classes OR HHAR_classes")#default="HHAR_classes",
    parser.add_argument("--data_folder", type=str, help="../uci_data/ or ../hhar_data/ ") #, default="../hhar_data/"
    parser.add_argument("--consistency", type=str,  help="kld or mse or criterion") #default="mse",
    
    parser.add_argument("--save_path", type=str, default="./checkpoint_saved/")
    parser.add_argument("--result_path", type=str, default="./result/")
    parser.add_argument("--augmentation", type=str, default="permute_timeShift_scale_noise",
                        help="negate_permute_timeShift_scale_noise")
    parser.add_argument("--device", type=str, default="mps",
                        help="cpu or mps or cuda:0")
    parser.add_argument("--oversample", type=bool, default=False,
                        help="apply oversampling or not?")

                        

    args = parser.parse_args()
    return args


def train(train_loader, test_loader, training_mode):
    # logger
    logger = starting_logs(args.dataset, training_mode,
                           args.result_path, args.data_percentage, args.consistency)
    
    num_clsTran_tasks = len(args.augmentation.split("_"))

    backbone_fe = getNetwork(args.dataset)
    backbone_temporal = net.cnn1d_temporal().to(args.device)
    
    if training_mode=="ssl":
        classifier = net.ssl_classifier(num_clsTran_tasks).to(args.device)
    else:
        classifier = net.classifier().to(args.device)

    # Average meters
    loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

    if training_mode == "ft":
        # load saved models
        # update: get the cp from the same folder
        chekpoint = torch.load(os.path.join(
            args.save_path, args.dataset, args.data_percentage, "ssl_checkpoint.pt"))
        backbone_fe.load_state_dict(chekpoint["fe"])

    elif training_mode not in ["ssl", "supervised", "s"]:
        print("Training mode not found!")

    network = nn.Sequential(backbone_fe, backbone_temporal, classifier)
    optimizer = optim.Adam(network.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay, betas=args.betas)

    best_f1 = 0
    best_acc = 0


    # training
    for e in range(args.nepoch):
        for sample in train_loader:
            # send data to device
            sample = to_device(sample, args.device)
            # print("sample")
            if training_mode == "ssl":
                # data pass to update(), return model
                losses, model = ssl_update(
                    backbone_fe, backbone_temporal, classifier, sample, optimizer, consistency=args.consistency)
                # cal metrics
                for key, val in losses.items():
                    loss_avg_meters[key].update(val, args.batchsize)

            elif training_mode != "ssl":  # supervised training or fine tuining
                losses, model = supervised_update(
                    backbone_fe, backbone_temporal, classifier, sample, optimizer)
                # cal metrics f1 acc rate
                for key, val in losses.items():
                    loss_avg_meters[key].update(val, args.batchsize)
            
            elif training_mode not in ["ssl", "supervised", "s", "ft"]:
                print("Training mode not found!")
                break

        # ft/supervised
        if training_mode != "ssl":
            y_pred, y_true = valid(
                test_loader, backbone_fe, backbone_temporal, classifier)
            acc_test, f1 = calc_results_per_run(y_pred, y_true)

            # save best model
            if f1 > best_f1:  # save best model based on best f1.
                best_f1 = f1
                best_acc = acc_test
                save_checkpoint(args.save_path, model,
                                args.dataset, training_mode, args.data_percentage)
                save_results(best_acc, best_f1)
                _save_metrics(y_pred, y_true, args.result_path, args.dataset, args.data_percentage,
                              args.training_mode, classes[args.classes])


        # logging
        logger.debug(f"print[Epoch : {e}/{args.nepoch}]")
        for key, val in loss_avg_meters.items():
            logger.debug(f"{key}\t: {val.avg:2.4f}")
            if training_mode != "ssl":
                acc_list.append(acc_test)
                f1_list.append(f1)
                logger.debug(
                    f"Acc:{acc_test:2.4f} \t F1:{f1:2.4f} (best: {best_f1:2.4f})")
        logger.debug(f"-------------------------------------")

        # save checkpoint
    if training_mode == "ssl":
        save_checkpoint(args.save_path, model, args.dataset, training_mode, args.data_percentage)
    
    #logger end
    logger.debug(f'Dataset: {args.dataset}, Training mode: {args.training_mode}, data %: {args.data_percentage}%, consistency: {args.consistency}')
    if training_mode != "ssl":
        logger.debug(f"Mean_Acc:{statistics.mean(acc_list):2.4f} \t Mean_F1:{statistics.mean(f1_list):2.4f} (best: {best_f1:2.4f})")
    logger.debug("=" * 45)

def valid(test_loader, feature_extractor, temporal_encoder, classifier):

    feature_extractor.eval()
    temporal_encoder.eval()
    classifier.eval()

    total_loss_ = []
    y_pred = np.array([])
    y_true = np.array([])

    with torch.no_grad():
        for data in test_loader:
            data_samples = to_device(data, args.device)  # sample to device/mps
            data = data_samples["sample_ori"].float()
            labels = data_samples["class_labels"].long()

            # forward pass
            features = feature_extractor(data)
            features = temporal_encoder(features)
            predictions = classifier(features)

            # compute loss
            loss = F.cross_entropy(predictions, labels)
            total_loss_.append(loss.item())
            # detach, to cpu
            # get the index of the max log-probability
            pred = predictions.detach().argmax(dim=1)

            y_pred = np.append(y_pred, pred.cpu().numpy().tolist())
            y_true = np.append(y_true, labels.data.cpu().numpy().tolist())

    return y_pred, y_true


def ssl_update(backbone_fe, backbone_temporal, classifier, samples, optimizer, consistency):

    # ====== Data =====================
    samples = to_device(samples, args.device)
    ori_data=samples["sample_ori"].float()
    data = samples["transformed_samples"].float()
    labels = samples["aux_labels"].long()

    optimizer.zero_grad()
    features = backbone_fe(data)

    features = features.flatten(1, 2)
    logits = classifier(features)
    
    # Cross-Entropy loss
    loss_1 = criterion(logits, labels)
    # KL divergence loss
    loss_2 = kl_loss(ori_data, data)
    # print("org_data: ", org_data.size(), "transformed data: ",data.size())
    # MSE loss
    loss_3 = mse_loss(ori_data, data)
    # print("org_data: ", org_data, "transformed data: ",data)

    if consistency == "kld":
        loss = loss_1+loss_2
        # print("1: ", loss_1.item(), "2: ", loss_2.item())
    elif consistency == "mse":
        loss = loss_1+loss_3  
        # print("1: ", loss_1.item(), "3: ", loss_3.item())
    else:
        loss = loss_1
    
    loss.backward()
    optimizer.step()

    model = [backbone_fe, backbone_temporal, classifier]
    # return loss.item()
    return {"Total_loss": loss.item()},model


def supervised_update(backbone_fe, backbone_temporal, classifier, samples, optimizer):
    # ====== Data =====================
    samples = to_device(samples, args.device)
    data = samples["sample_ori"].float()
    labels = samples["class_labels"].long()

    optimizer.zero_grad()
    features = backbone_fe(data)
    features = backbone_temporal(features)
    logits = classifier(features)

    # Cross-Entropy loss
    loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()

    model=[backbone_fe, backbone_temporal, classifier]
    # return loss.item()
    return {"Total_loss": loss.item()}, model


# get Network - ssl/supervised
# update: to device
def getNetwork(dataset):
    if dataset=="HHAR":
        backbone_fe = net.cnnNetwork_HHAR().to(args.device)
    elif dataset=="UCI":
        backbone_fe = net.cnnNetwork_UCI().to(args.device)
    elif dataset=="HAPT":
        backbone_fe = net.cnnNetwork_HAPT().to(args.device) 
    else:
        print("Dataset not found!")
    return backbone_fe

def calc_results_per_run(pred_labels, true_labels):
    acc, f1 = _calc_metrics(pred_labels, true_labels, classes[args.classes])
    return acc, f1


def save_results(best_acc, best_f1):
    metrics = {"accuracy": [], "f1_score": []}
    run_metrics = {"accuracy": best_acc, "f1_score": best_f1}
    df = pd.DataFrame(columns=["acc", "f1"])
    df.loc[0] = [best_acc, best_f1]

    for (key, val) in run_metrics.items():
        metrics[key].append(val)
    
    create_folder(args.result_path, args.dataset, args.data_percentage, args.training_mode)

    scores_save_path = os.path.join(
        args.result_path, args.dataset, args.data_percentage, args.training_mode, "scores.xlsx")
    df.to_excel(scores_save_path, index=False)


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    train_loader, test_loader = load_data(args.data_folder, args.training_mode, args.data_percentage,
                                          augmentation=args.augmentation, oversample=args.oversample)
    # Train model
    train(train_loader, test_loader, args.training_mode)


