import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from logger import Logger
from model.Unet import Unet
from model.ResNet import ResNet, BasicBlock
from model.CascadeModel import CascadeModel
from dataset import RMLDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='train program for unet-resnet model',
)

parser.add_argument('-d', '--dataset', default="./data/data_classifier_time_stft.pkl",
                    help="dataset for classifier", type=str)
parser.add_argument("-cm", '--cls_model', default="checkpoints/resnet.pth", type=str)
parser.add_argument("-b", '--batch_size', default=512, type=int)
parser.add_argument("-lr", '--learning_rate', default=1e-4, type=float)
parser.add_argument("-e", '--epochs', default=100, type=int)

args = parser.parse_args()


logger = Logger("train4unet_resnet.log")
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


classifier_model = ResNet(BasicBlock, [2, 2, 2], num_classes=10)
denoise_model = Unet()
cas_model = CascadeModel(denoise_model, classifier_model)

cas_model.modelB.load_state_dict(torch.load(args.cls_model,
                                                 weights_only=True))

for param in cas_model.modelA.parameters():  # 降噪模块
    param.requires_grad = True

for param in cas_model.modelB.parameters():  # 分类模块
    param.requires_grad = False

writer = SummaryWriter(f"logs/unet-resnet")


lr = args.learning_rate
epochs = args.epochs

logger.info("开始加载数据")
SNR_LEVEL = list(range(-10, 22, 2))
dataset = RMLDataset(args.dataset, snr_list=SNR_LEVEL, seed=2025)
train_dataset, test_dataset = dataset.split_data(n_train=0.75)
logger.info("数据加载完毕")

cas_model = cas_model.to(device)

train_data = DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=False)
test_data = DataLoader(test_dataset, args.batch_size, shuffle=True, drop_last=False)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cas_model.parameters()), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)

train_dataset_len = len(train_dataset)
test_dataset_len = len(test_dataset)

max_acc = 0.0

for epoch in tqdm(range(epochs)):
    cas_model.train()
    cas_model.modelB.eval()
    running_loss = 0.0
    for x, y in train_data:
        x, y = x.to(device), y.to(device)
        # 更新参数需要清空梯度
        optimizer.zero_grad()
        out = cas_model(x)
        loss = loss_func(out, y)
        loss.backward()
        # 更新参数
        optimizer.step()
        with torch.no_grad():
            running_loss += loss.item()
    epoch_loss = running_loss / len(train_data)
    logger.info(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.7f}")
    writer.add_scalar(f"train_loss", epoch_loss, epoch)

    val_loss = 0.0
    accuracy = 0.0
    with torch.no_grad():
        cas_model.eval()
        for x, y in test_data:
            x, y = x.to(device), y.to(device)
            y_pred = cas_model(x)
            val_loss += loss_func(y, y_pred).item()
            accuracy += (y.argmax(dim=1) == y_pred.argmax(dim=1)).sum()  # 计算正确的个数
        accuracy = accuracy / test_dataset_len
        val_loss = val_loss / len(test_data)
        logger.info('Test Accuracy: {:.5f}'.format(accuracy))
        writer.add_scalar(f'Accuracy/test', accuracy, epoch)

    logger.info('Learning rate: {}'.format(scheduler.get_last_lr()))
    scheduler.step(val_loss)

    if accuracy > max_acc:
        max_acc = accuracy
        torch.save(cas_model.state_dict(), f"checkpoints/unet_resnet.pth")