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
import torch.nn.functional as F
from utils import AverageMeter, to_device, _save_metrics, starting_logs, save_checkpoint, _calc_metrics

# update with dataset
classes = ['WALKING', 'WALKING_UPSTAIRS',
           'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
criterion = nn.CrossEntropyLoss()


def get_args():
    parser = argparse.ArgumentParser()
    # ===================parameters===========================
    parser.add_argument('--nepoch', type=int, default=30)  # 50
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)  # 0.0003
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--betas', type=float, default=(0.9, 0.999))
    parser.add_argument('--seed', type=int, default=10)
    # ===================settings===========================
    parser.add_argument('--data_folder', type=str, default='../uci_data/')
    parser.add_argument('--data_percentage', type=str, default='100')
    parser.add_argument('--save_path', type=str, default='./checkpoint_saved/')
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--dataset', type=str,
                        default='UCI', help='HAPT OR HHAR')
    parser.add_argument('--training_mode', type=str, default='supervised',
                        help='Modes of choice: supervised, ssl, ft')
    parser.add_argument('--augmentation', type=str, default='permute_timeShift_scale_noise',
                        help='negate_permute_timeShift_scale_noise')
    parser.add_argument('--device', type=str, default='mps',
                        help='cpu or mps or cuda:0')
    parser.add_argument('--oversample', type=bool, default=False,
                        help='apply oversampling or not?')

    args = parser.parse_args()
    return args


def train(train_loader, test_loader, training_mode):
    # logger
    logger = starting_logs(args.dataset, training_mode,
                           args.result_path, args.data_percentage)

    # get Network - ssl/supervised
    # update: to device
    backbone_fe = net.cnnNetwork().to(args.device)
    backbone_temporal = net.cnn1d_temporal().to(args.device)
    classifier = net.classifier().to(args.device)

    # Average meters
    loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

    if training_mode == "ft":
        # load saved models
        # update: get the cp from the same folder
        chekpoint = torch.load(os.path.join(
            args.save_path, args.dataset, 'ssl_checkpoint.pt'))
        backbone_fe.load_state_dict(chekpoint["fe"])

    elif training_mode not in ["ssl", "supervised"]:
        print("Training mode not found!")

    network = nn.Sequential(backbone_fe, backbone_temporal, classifier)
    optimizer = optim.Adam(network.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay, betas=(0.9, 0.99))

    best_f1 = 0
    best_acc = 0

    # training
    for e in range(args.nepoch):

        for sample in train_loader:
            # send data to device
            sample = to_device(sample, args.device)

            if training_mode == "ssl":
                # data pass to update(), return model
                losses, model = ssl_update(
                    backbone_fe, backbone_temporal, classifier, sample, optimizer)
                # cal metrics
                for key, val in losses.items():
                    loss_avg_meters[key].update(val, args.batchsize)

            elif training_mode != "ssl":  # supervised training or fine tuining
                losses, model = surpervised_update(
                    backbone_fe, backbone_temporal, classifier, sample, optimizer)
                # cal metrics f1 acc rate
                for key, val in losses.items():
                    loss_avg_meters[key].update(val, args.batchsize)

        # testing
        # update: indentation
        if training_mode != "ssl":
            y_pred, y_true = valid(
                test_loader, backbone_fe, backbone_temporal, classifier)
            acc_test, f1 = calc_results_per_run(y_pred, y_true)

            # save best model
            if f1 > best_f1:  # save best model based on best f1.
                best_f1 = f1
                best_acc = acc_test
                save_checkpoint(args.save_path, model,
                                args.dataset, training_mode)
                save_results(best_acc, best_f1)
                _save_metrics(y_pred, y_true, args.result_path,
                              args.training_mode, classes)

        # logging
        logger.debug(f'print[Epoch : {e}/{args.nepoch}]')
        for key, val in loss_avg_meters.items():
            logger.debug(f'{key}\t: {val.avg:2.4f}')
            if training_mode != "ssl":
                logger.debug(
                    f'Acc:{acc_test:2.4f} \t F1:{f1:2.4f} (best: {best_f1:2.4f})')
        logger.debug(f'-------------------------------------')

        # save checkpoint
    if training_mode == "ssl":
        save_checkpoint(args.save_path, model, args.dataset, training_mode)


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

            data = data_samples['sample_ori'].float()
            labels = data_samples['class_labels'].long()

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

        # trg_loss = torch.tensor(total_loss_).mean()  # average loss

    return y_pred, y_true


def ssl_update(backbone_fe, backbone_temporal, classifier, samples, optimizer):

    # ====== Data =====================
    samples = to_device(samples, args.device)
    data = samples["transformed_samples"].float()
    labels = samples["aux_labels"].long()

    optimizer.zero_grad()
    features = backbone_fe(data)

    features = features.flatten(1, 2)
    logits = classifier(features)

    # Cross-Entropy loss
    loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()

    # return loss.item()
    return {'Total_loss': loss.item()}, \
        [backbone_fe, backbone_temporal, classifier]


def surpervised_update(backbone_fe, backbone_temporal, classifier, samples, optimizer):
    # ====== Data =====================
    samples = to_device(samples, args.device)
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

    for (key, val) in run_metrics.items():
        metrics[key].append(val)

    scores_save_path = os.path.join(
        args.result_path, args.training_mode, "scores.xlsx")
    df.to_excel(scores_save_path, index=False)


def create_folder(folder_name, train_mode):
    log_dir = os.path.join(".", train_mode, folder_name)
    os.makedirs(log_dir, exist_ok=True)


if __name__ == '__main__':
    args = get_args()
    print(args.training_mode)
    torch.manual_seed(args.seed)
    train_loader, test_loader = load_data(args.data_folder, args.training_mode, args.data_percentage,
                                          augmentation=args.augmentation, oversample=args.oversample)

    num_clsTran_tasks = len(args.augmentation.split("_"))

    # Train model
    train(train_loader, test_loader, args.training_mode)
