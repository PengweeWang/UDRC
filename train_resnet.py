import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import RMLDataset
from model.ResNet import ResNet, BasicBlock
from torch.utils.tensorboard import SummaryWriter
from logger import Logger

parser = argparse.ArgumentParser(
    description='train program for resnet model',
)

parser.add_argument('-d', '--dataset', default="./data/data_classifier_time_stft.pkl",
                    help="dataset for classifier", type=str)
parser.add_argument("-b", '--batch_size', default=512, type=int)
parser.add_argument("-lr", '--learning_rate', default=2e-5, type=float)
parser.add_argument("-e", '--epochs', default=100, type=int)
args = parser.parse_args()

logger = Logger("train4classifier.log")

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tensorboard
writer = SummaryWriter("logs/ResNet")
# dataset
SNR_LEVEL = list(range(6, 22, 2))

logger.info("开始加载数据")
dataset = RMLDataset(args.dataset, snr_list=SNR_LEVEL, seed=2025)
train_dataset, test_dataset = dataset.split_data(n_train=0.75)
logger.info("数据加载完毕")

# dataloader
train_dataset_len = len(train_dataset)
test_dataset_len = len(test_dataset)
train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
test_data = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

# param
n = "0" # 记录训练参数及次数
learn_rate = args.learning_rate
epoch = args.epochs
model = ResNet(BasicBlock, [2, 2, 2], num_classes=10).to(device)

# loss func
loss_func = nn.CrossEntropyLoss()
loss_func.to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

# lr scheduler
lr_scheduler =torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5)

epoch_loss = 0.0
for i in range(epoch):
    running_loss = 0.0
    model.train()
    for x, y in train_data:
        x, y = x.to(device), y.to(device)
        y_pred = model(x) # 预测
        loss = loss_func(y_pred, y) # 计算损失
        # 反向传播
        optimizer.zero_grad() # 反向传播前清空梯度
        loss.backward() # 误差反向传播
        # 更新参数
        optimizer.step()
        with torch.no_grad():
            running_loss +=loss.item()
    epoch_loss = running_loss / len(train_data) # 计算所有数据训练一轮后的平均误差
    logger.info('Epoch: {}, Loss: {:.5f}'.format(i, epoch_loss))
    writer.add_scalar(f'Loss/train-{n}', epoch_loss, i)

    # 计算验证集损失
    val_loss = 0.0
    with torch.no_grad():
        accuracy = 0.0
        model.eval()

        for x, y in test_data:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            val_loss +=loss_func(y, y_pred).item()
            accuracy += (y.argmax(dim=1) == y_pred.argmax(dim=1)).sum() # 计算正确的个数
        accuracy = accuracy / test_dataset_len
        val_loss = val_loss / len(test_data)
        logger.info('Test Accuracy: {:.5f}'.format(accuracy))
        writer.add_scalar(f'Accuracy/test{n}', accuracy, i)
    logger.info('Learning rate: {}'.format(lr_scheduler.get_last_lr()))
    lr_scheduler.step() # note


torch.save(model.state_dict(),
           f'./checkpoints/resnet.pth')





















