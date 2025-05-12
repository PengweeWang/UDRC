import argparse
import torch
from tqdm import tqdm
from model.ResNet import ResNet, BasicBlock
from model.Unet import Unet
from model.CascadeModel import CascadeModel
from dataset import RMLDataset, BaseRadioDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(
    description='test program for all model',
)

parser.add_argument('-d', '--dataset', default="./data/data_classifier_time_stft.pkl",
                    help="dataset for classifier", type=str)
parser.add_argument("-b", '--batch_size', default=512, type=int)
parser.add_argument("-t", '--type', choices=['unet-noise', 'unet-signal', 'unet-resnet', 'resnet'], type=str)
parser.add_argument("-m", '--model', type=str)
parser.add_argument("-cm", '--cls_model', default="./checkpoints/resnet.pth", type=str, help='classifier model path')

args = parser.parse_args()
if args.type in ['unet-noise', 'unet-signal'] and not args.cls_model:
    parser.error(f"when testing {args.type} model you need to provide --cls_model param (the path of classifier model)")



def test(testModel, testDataset, SNR_list: list, batch_size = 64) -> tuple:
    testModel.eval()
    testModel.to(device)
    y_list = []
    y_pre_list = []
    accuracy_list = []
    for snr in tqdm(SNR_list):
        test_dataset = testDataset.get_snr_data(snr)
        test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        with torch.no_grad():
            accuracy = 0.0
            for x, y in test_data:
                x, y = x.to(device), y.to(device)
                y_pred = testModel(x)
                y_list.append(y.argmax(dim=1).cpu().numpy())
                y_pre_list.append(y_pred.argmax(dim=1).cpu().numpy())
                accuracy_num = (y.argmax(dim=1) == y_pred.argmax(dim=1)).sum()
                accuracy += accuracy_num.item()
            accuracy_list.append(accuracy / len(test_dataset))
    y_list = np.concatenate(y_list)
    y_pre_list = np.concatenate(y_pre_list)

    cm = confusion_matrix(y_list, y_pre_list, normalize='true')
    acc_all = (y_list == y_pre_list).sum()/len(y_list)
    return acc_all, accuracy_list, cm

def test_for_noise(testModel, testDataset, SNR_list: list, batch_size = 64) -> tuple:
    testModel.eval()
    testModel.to(device)
    y_list = []
    y_pre_list = []
    accuracy_list = []
    for snr in tqdm(SNR_list):
        test_dataset = testDataset.get_snr_data(snr)
        test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        with torch.no_grad():
            accuracy = 0.0
            for x, y in test_data:
                x, y = x.to(device), y.to(device)
                denoised_x = x - testModel.denoise(x)
                y_pred = testModel.classifier(denoised_x)
                y_list.append(y.argmax(dim=1).cpu().numpy())
                y_pre_list.append(y_pred.argmax(dim=1).cpu().numpy())
                accuracy_num = (y.argmax(dim=1) == y_pred.argmax(dim=1)).sum()
                accuracy += accuracy_num.item()
            accuracy_list.append(accuracy / len(test_dataset))
    y_list = np.concatenate(y_list)
    y_pre_list = np.concatenate(y_pre_list)

    cm = confusion_matrix(y_list, y_pre_list, normalize='true')
    acc_all = (y_list == y_pre_list).sum()/len(y_list)
    return acc_all, accuracy_list, cm

def visualization(snr_list, acc_list, mode, cm):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # 绘制正确率曲线
    plt.plot(snr_list, acc_list,
             marker='o',
             markersize=8,
             linewidth=2,
             color='#2c7bb6',
             markerfacecolor='#d7191c',
             markeredgecolor='white',
             markeredgewidth=1.5)


    plt.fill_between(snr_list, acc_list,
                     color='#abd9e9',
                     alpha=0.4)


    plt.xlim([-10, 20])
    plt.ylim([0.0, 1.01])
    plt.xticks(np.arange(-10, 21, 5), fontsize=12)
    plt.yticks(np.arange(0, 1.05, 0.1), fontsize=12)
    plt.xlabel('SNR (dB)', fontsize=14, labelpad=10)
    plt.ylabel('Accuracy', fontsize=14, labelpad=10)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)

    max_acc = max(acc_list)
    max_snr = snr_list[acc_list.index(max_acc)]
    plt.annotate(f'Max Accuracy: {max_acc:.4f}',
                 (max_snr, max_acc),
                 xytext=(-50, -60),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                 fontsize=12)

    avg_acc = np.mean(acc_list)
    plt.axhline(avg_acc, color='#fdae61', linestyle='--', linewidth=1.5)
    plt.text(18, avg_acc + 0.02, f'Avg: {avg_acc:.4f}', ha='right', fontsize=12)

    plt.tight_layout()
    plt.figure()
    ax = sns.heatmap(cm, annot=True, fmt='.2f', cmap="Blues", xticklabels=mode, yticklabels=mode)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    SNR_LEVEL = list(range(-10, 22, 2))
    dataset = RMLDataset(args.dataset, snr_list=SNR_LEVEL, seed=2025)
    _, TestDataset = dataset.split_data(n_train=0.75)
    mode = dataset.modulation_mode
    acc_list_B = None
    cm_B = None
    if args.type == "resnet":
        model = ResNet(BasicBlock, [2, 2, 2], 10)
        model.load_state_dict(torch.load(args.model, weights_only=True))
        acc_all_B, acc_list_B, cm_B = test(model, TestDataset, SNR_LEVEL)

    if args.type == "unet-resnet":
        classifier_model = ResNet(BasicBlock, [2, 2, 2], num_classes=10)
        denoise_model = Unet()
        model = CascadeModel(denoise_model, classifier_model)
        model.load_state_dict(torch.load(args.model, weights_only=True))
        acc_all_B, acc_list_B, cm_B = test(model, TestDataset, SNR_LEVEL)


    if args.type == "unet-noise":
        classifier_model = ResNet(BasicBlock, [2, 2, 2], num_classes=10)
        denoise_model = Unet()
        model = CascadeModel(denoise_model, classifier_model)
        model.modelA.load_state_dict(torch.load(args.model, weights_only=True))
        model.modelB.load_state_dict(torch.load(args.cls_model, weights_only=True))
        acc_all_B, acc_list_B, cm_B = test_for_noise(model, TestDataset, SNR_LEVEL)


    if args.type == "unet-signal":
        classifier_model = ResNet(BasicBlock, [2, 2, 2], num_classes=10)
        denoise_model = Unet()
        model = CascadeModel(denoise_model, classifier_model)
        model.modelA.load_state_dict(torch.load(args.model, weights_only=True))
        model.modelB.load_state_dict(torch.load(args.cls_model, weights_only=True))
        acc_all_B, acc_list_B, cm_B = test(model, TestDataset, SNR_LEVEL)
    if acc_list_B is not None and cm_B is not None:
        visualization(SNR_LEVEL, acc_list_B, mode, cm_B)

