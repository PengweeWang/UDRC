import pickle as pk
import numpy as np
import torch
from torch.utils.data import Dataset



class BaseDataset(Dataset):
    def __init__(self, x, label):
        self.x = x
        self.label = label

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.label[idx]

class BaseModeDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def get_snr_data(self, snr):
        return self.data_dict[snr]


class BaseRadioDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        keys = list(self.data_dict.keys())
        self.modulation_mode = list(dict.fromkeys(i[0] for i in keys))  # 自动去除重复项
        self.n_mode = len(self.modulation_mode)
        self.snr_levels = sorted(list(set([i[1] for i in list(keys)])))
        self.x, self.label = self._init_data()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.label[idx]

    def _init_data(self):
        x = []
        label = []
        for snr_level in self.snr_levels:
            for s in range(self.n_mode):
                key = (self.modulation_mode[s], snr_level)
                data_snr_s = self.data_dict[key]
                data_snr_len = len(data_snr_s)
                x.append(data_snr_s)
                label.append([np.eye(self.n_mode)[s] for _ in range(data_snr_len)])
        x = np.vstack(x)
        label = np.vstack(label)
        x_sample_shape = x.shape[1:]
        x = torch.tensor(x, dtype=torch.float32).reshape(len(x), 1, x_sample_shape[0], x_sample_shape[1])
        label = torch.tensor(label, dtype=torch.float32)
        return x, label

    def get_snr_data(self, snr):
        x = []
        label = []
        for s in range(self.n_mode):
            key = (self.modulation_mode[s], snr)
            data_snr_s = self.data_dict[key]
            data_snr_len = len(data_snr_s)
            x.append(data_snr_s)
            label.append([np.eye(self.n_mode)[s] for _ in range(data_snr_len)])
        x = np.vstack(x)
        label = np.vstack(label)
        x_sample_shape = x.shape[1:]
        x = torch.tensor(x, dtype=torch.float32).reshape(len(x), 1, x_sample_shape[0], x_sample_shape[1])
        label = torch.tensor(label, dtype=torch.float32)
        return BaseDataset(x, label)

    def get_mode_data(self, mode):
        x = []
        label = []
        for snr in self.snr_levels:
            s = self.modulation_mode.index(mode)
            key = (mode, snr)
            data_snr_s = self.data_dict[key]
            data_snr_len = len(data_snr_s)
            x.append(data_snr_s)
            label.append([np.eye(self.n_mode)[s] for _ in range(data_snr_len)])
        x = np.vstack(x)
        label = np.vstack(label)
        x_sample_shape = x.shape[1:]
        x = torch.tensor(x, dtype=torch.float32).reshape(len(x), 1, x_sample_shape[0], x_sample_shape[1])
        label = torch.tensor(label, dtype=torch.float32)
        return BaseDataset(x, label)

    def get_mode_snr_data(self, mode):
        snr_dataset_dict = {}
        for snr in self.snr_levels:
            s = self.modulation_mode.index(mode)
            key = (mode, snr)
            x = np.array(self.data_dict[key])
            data_snr_len = len(x)
            label = np.array([np.eye(self.n_mode)[s] for _ in range(data_snr_len)])
            x_sample_shape = x.shape[1:]
            x = torch.tensor(x, dtype=torch.float32).reshape(len(x), 1, x_sample_shape[0], x_sample_shape[1])
            label = torch.tensor(label, dtype=torch.float32)
            snr_dataset_dict[snr] = BaseDataset(x, label)
        return BaseModeDataset(snr_dataset_dict)



class RMLDataset:
    def __init__(self, datafile, encoding="ASCII", snr_list=None, seed=None):
        """
        保证设置相同的seed时，无论snr_list以及low_snr_ratio的值，产生的测试集不在训练集中
        :param datafile: 数据集文件
        :param encoding: 数据集编码的格式
        :param snr_list: 信噪比序列
        :param seed: 随机数种子
        """
        fd = open(datafile, 'rb')
        self.data = pk.load(fd, encoding=encoding)
        keys = list(self.data.keys())
        self.modulation_mode = list(dict.fromkeys(i[0] for i in keys))  # 自动去除重复项
        self.n_mode = len(self.modulation_mode)
        self.snr_levels = sorted(list(set([i[1] for i in list(keys)])))
        if snr_list is None:
            snr_list = self.snr_levels
        self.snr_list = snr_list
        self.seed = seed




    def split_data(self, n_train: float = 0.8):
        if not (0 <= n_train <= 1):
            raise ValueError("n_train should be a float between 0 and 1")

        train_dict = {}
        test_dict = {}

        if self.seed is not None:
            np.random.seed(self.seed)

        for snr in self.snr_list:
            for mode in self.modulation_mode:
                data_mode_snr = self.data[(mode, snr)]
                data_mode_snr_len = len(data_mode_snr)
                train_samples_len = int(data_mode_snr_len*n_train)
                # 打乱data_mode_snr
                np.random.shuffle(data_mode_snr)
                train_dict[(mode, snr)] = data_mode_snr[:train_samples_len]
                test_dict[(mode, snr)] = data_mode_snr[train_samples_len:]

        return BaseRadioDataset(train_dict), BaseRadioDataset(test_dict)


class BaseDenoiseDataset(Dataset):
    def __init__(self, x, label, snr):
        self.x = x
        self.label = label
        self.snr = snr

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.label[idx], self.snr[idx]

class RMLDenoiseDataset:
    def __init__(self, clean_data_path, noise_data_path):
        clean_data_fd = open(clean_data_path, "rb")
        noise_data_fd = open(noise_data_path, "rb")

        self.clean_data = pk.load(clean_data_fd)
        self.noise_data = pk.load(noise_data_fd)

        clean_data_fd.close()
        noise_data_fd.close()

        keys = list(self.noise_data.keys())
        self.modulation_mode = list(dict.fromkeys(i[0] for i in keys))  # 自动去除重复项
        self.n_mode = len(self.modulation_mode)
        self.snr_levels = sorted(list(set([i[1] for i in list(keys)])))

        self.x, self.label, self.label_mode = self._init_data()



    def _init_data(self):
        x = []
        label = []
        mode_list = []
        for snr in self.snr_levels:
            # for mode in self.modulation_mode:
            for s in range(self.n_mode):
                mode = self.modulation_mode[s]
                key = (mode, snr)
                x_data = self.noise_data[key]
                x_data_len = len(x_data)
                label_data = self.clean_data[key]
                x.append(x_data)
                label.append(label_data)
                mode_list.append([np.eye(self.n_mode)[s] for _ in range(x_data_len)])

        x = np.vstack(x)
        label = np.vstack(label)
        mode_list = np.vstack(mode_list)

        x_sample_shape = x.shape[1:]
        label_sample_shape = label.shape[1:]

        x = torch.tensor(x, dtype=torch.float32).reshape(len(x), 1, x_sample_shape[0], x_sample_shape[1])
        label = torch.tensor(label, dtype=torch.float32).reshape(len(label), 1, label_sample_shape[0], label_sample_shape[1])
        mode_list = torch.tensor(mode_list, dtype=torch.float32)
        return x, label, mode_list


    def get_dataset(self):
        return BaseDataset(self.x, self.label)

    def get_dataset_with_snr(self):
        return BaseDenoiseDataset(self.x, self.label, self.label_mode)

    def split_data(self, n_train = 0.8):
        n_train_samples = int(n_train*len(self.x))

        idx = np.random.permutation(len(self.x))

        shuffled_x = self.x[idx]
        shuffled_label = self.label[idx]

        train_x = shuffled_x[:n_train_samples]
        train_label = shuffled_label[:n_train_samples]

        test_x = shuffled_x[n_train_samples:]
        test_label = shuffled_label[n_train_samples:]

        return BaseDataset(train_x, train_label), BaseDataset(test_x, test_label)






    


