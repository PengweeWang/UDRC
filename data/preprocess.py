import pickle
import numpy as np
from utils import preprocessing
import pickle
import sys
from pathlib import Path

file_path = Path(sys.argv[1])
file_name = file_path.name.split('.')[0]

fd = open(file_path, 'rb')
data = pickle.load(fd)
keys = list(data.keys())
modulation_mode = list(dict.fromkeys(i[0] for i in keys))  # 自动去除重复项
print(modulation_mode)
snr_levels = sorted(list(set([i[1] for i in list(keys)])))
print(snr_levels)
data_stft = {}
for modulation_mode_i in modulation_mode:
    for snr_level in snr_levels:
        data_i = data[(modulation_mode_i, snr_level)]
        data_i_stft = []
        for data_s in data_i:
            data_s_stft = preprocessing(data_s, target_shape=(64, 64))
            data_i_stft.append(data_s_stft)
        data_stft[(modulation_mode_i, snr_level)] = data_i_stft
    print('{} done'.format(modulation_mode_i))

pickle.dump(data_stft, open(f'data/{file_name}_stft.pkl', 'wb'))
print(f'{file_name}_stft.pkl done')