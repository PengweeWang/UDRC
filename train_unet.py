import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from logger import Logger
from model.Unet import  Unet
from dataset import RMLDenoiseDataset
from torch.utils.tensorboard import SummaryWriter
import argparse

parser = argparse.ArgumentParser(
    description='train program for 2 type unet denoise model',
)

parser.add_argument("-t", '--type', choices=['noise', 'signal'])
parser.add_argument('-cdp', '--clean_data_path', default="./data/data_denoise_y_stft.pkl",
                    help="Clean time-frequency graph data path", type=str)
parser.add_argument('-ndp', '--noise_data_path', default="./data/data_denoise_x_stft.pkl",
                    help="Noise time-frequency graph data path", type=str)
parser.add_argument("-b", '--batch_size', default=512, type=int)
parser.add_argument("-lr", '--learning_rate', default=1e-4, type=float)
parser.add_argument("-e", '--epochs', default=100, type=int)



def save_model(model_name, model):
    torch.save(model.state_dict(),
               f'./checkpoints/{model_name}.pth')


def train_for_signal(clean_data_path, noise_data_path, batch_size=512, epochs=100, lr=1e-4):
    logger = Logger("train4denoise.log")
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tensorboard
    writer = SummaryWriter("logs/denoise_unet_noise")

    logger.info("开始加载数据")
    denoise_dataset = RMLDenoiseDataset(clean_data_path,
                                        noise_data_path).get_dataset_with_snr()
    dataset_len = len(denoise_dataset)
    logger.info("数据加载完毕")

    model = Unet().to(device)

    train_data = DataLoader(denoise_dataset, batch_size, shuffle=True, drop_last=False)

    # param
    loss_func = nn.MSELoss()
    # loss_func = HybridLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y, _ in train_data:
            x, y = x.to(device), y.to(device)
            # 更新参数需要清空梯度
            optimizer.zero_grad()
            out = model(x)
            loss = loss_func(out, y)
            loss.backward()
            # 更新参数
            optimizer.step()

            with torch.no_grad():
                running_loss += loss.item()
        epoch_loss = running_loss / dataset_len
        logger.info(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.7f}")
        writer.add_scalar(f"loss", epoch_loss, epoch)
        scheduler.step(epoch_loss)

    torch.save(model.state_dict(),
               f'./checkpoints/denoise_unet_noise.pth')

def train_for_noise(clean_data_path, noise_data_path, batch_size=512, epochs=100, lr=1e-4):
    logger = Logger("train4denoise.log")
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tensorboard
    writer = SummaryWriter("logs/denoise_unet_signal")

    logger.info("开始加载数据")
    denoise_dataset = RMLDenoiseDataset(clean_data_path,
                                        noise_data_path).get_dataset_with_snr()
    dataset_len = len(denoise_dataset)
    logger.info("数据加载完毕")

    model = Unet().to(device)

    train_data = DataLoader(denoise_dataset, batch_size, shuffle=True, drop_last=False)

    # param
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y, _ in train_data:
            x, y = x.to(device), y.to(device)
            # 更新参数需要清空梯度
            optimizer.zero_grad()
            out = model(x)
            loss = loss_func(out, x - y)
            loss.backward()
            # 更新参数
            optimizer.step()

            with torch.no_grad():
                running_loss += loss.item()
        epoch_loss = running_loss / dataset_len
        logger.info(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.7f}")
        writer.add_scalar(f"loss", epoch_loss, epoch)
        scheduler.step(epoch_loss)

    torch.save(model.state_dict(),
               f'./checkpoints/denoise_unet_signal.pth')



if __name__=="__main__":
    args = parser.parse_args()
    if args.type == "signal":
        train_for_signal(args.clean_data_path, args.noise_data_path, args.batch_size, args.epochs, args.learning_rate)
    elif args.type == "noise":
        train_for_noise(args.clean_data_path, args.noise_data_path, args.batch_size, args.epochs, args.learning_rate)




