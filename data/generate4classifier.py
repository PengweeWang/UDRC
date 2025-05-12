# 生成不同参数，不同信噪比的雷达脉内调制信号
# 每种信号信噪比range(-6, 20), 每个信噪比800条数据
# 13*800*10 = 104000条数据
from radio import *
import numpy as np
from config import SAMPLE_NUM, SAMPLE_RATE, BANDWIDTH_MAX, SAMPLE_DURATION
from utils import awgn, bark_code, generate_nfsk_sequence
import random
import pickle
snr_list = list(range(-10, 22, 2))

data = {}

for snr in snr_list:
    lfm_snr = []
    cw_snr = []
    vtfm_snr = []
    sfm_snr = []
    eqfm_snr = []
    fsk2_snr = []
    fsk4_snr = []
    bpsk_snr = []
    frank_snr = []
    lfm_bpsk_snr = []
    for i in range(400):
        # LFM
        lfm_f0 = random.uniform(SAMPLE_RATE * 0.05, SAMPLE_RATE * 0.45) # 0.05~0.45
        lfm_f1 = random.uniform(0.05 * SAMPLE_RATE, 0.5 * SAMPLE_RATE - lfm_f0) + lfm_f0 # 0.05~0.5, 带宽最小为0.05
        _, lfm_signal = generate_lfm_signal(duration=SAMPLE_DURATION, sample_rate=SAMPLE_RATE, f0=lfm_f0, f1=lfm_f1)
        lfm_signal = awgn(lfm_signal, snr)
        lfm_snr.append(lfm_signal)
        # CW
        cw_f0 = random.uniform(SAMPLE_RATE * 0.05, SAMPLE_RATE * 0.45)
        _, cw_signal = generate_cw_signal(duration=SAMPLE_DURATION, sample_rate=SAMPLE_RATE, f0=cw_f0)
        cw_signal = awgn(cw_signal, snr)
        cw_snr.append(cw_signal)
        # VTFM
        vtfm_f0 = random.uniform(SAMPLE_RATE * 0.1, SAMPLE_RATE * 0.4)
        vtfm_B = random.uniform(0.05 * SAMPLE_RATE, 0.45 * SAMPLE_RATE - vtfm_f0)
        _, vtfm_signal = generate_vfm_signal(duration=SAMPLE_DURATION, sample_rate=SAMPLE_RATE, f0=vtfm_f0, B=vtfm_B)
        vtfm_signal = awgn(vtfm_signal, snr)
        vtfm_snr.append(vtfm_signal)
        # SFM
        sfm_f0 = random.uniform(SAMPLE_RATE * 0.1, SAMPLE_RATE * 0.4)
        sfm_B = random.uniform(0.1 * SAMPLE_RATE, min(sfm_f0, 0.5*SAMPLE_RATE-sfm_f0))*2
        sfm_mo = random.uniform(1, 3) # * 周期参数大于1
        _, sfm_signal = generate_sfm_signal(duration=SAMPLE_DURATION, sample_rate=SAMPLE_RATE, f0=sfm_f0, B=sfm_B, mo=sfm_mo)
        sfm_signal = awgn(sfm_signal, snr)
        sfm_snr.append(sfm_signal)
        # EQFM
        eqfm_f0 = random.uniform(SAMPLE_RATE * 0.1, SAMPLE_RATE * 0.4)
        eqfm_B = random.uniform(0.05 * SAMPLE_RATE, 0.45 * SAMPLE_RATE - eqfm_f0)
        _, eqfm_signal = generate_eqfm_signal(duration=SAMPLE_DURATION, sample_rate=SAMPLE_RATE, f0=eqfm_f0, B=eqfm_B)
        eqfm_signal = awgn(eqfm_signal, snr)
        eqfm_snr.append(eqfm_signal)
        # 2FSK
        fsk2_deltaf = random.uniform(0.1 * SAMPLE_RATE, 0.2 * SAMPLE_RATE)
        fsk2_list = [0.25 * SAMPLE_RATE - fsk2_deltaf, 0.25 * SAMPLE_RATE + fsk2_deltaf]
        fsk2_seq_n = random.randint(8, 16) # 码元宽度N/16 ~ N/8
        fsk2_seq = generate_nfsk_sequence(fsk2_seq_n, 2)
        _, fsk2_signal = generate_fsk_signal(sample_num=SAMPLE_NUM, sample_rate=SAMPLE_RATE, freq_list=fsk2_list,
                                            sequence=fsk2_seq)
        fsk2_signal = awgn(fsk2_signal, snr)
        fsk2_snr.append(fsk2_signal)
        # 4FSK
        fsk4_edgef = random.uniform(0.01 * SAMPLE_RATE, 0.1 * SAMPLE_RATE) # 两侧宽度
        fsk4_deltaf = (0.5 * SAMPLE_RATE - fsk4_edgef*2)/3 # 4fsk频率间间隔
        fsk4_list = [fsk4_edgef, fsk4_edgef + fsk4_deltaf, fsk4_edgef + 2*fsk4_deltaf, fsk4_edgef + 3*fsk4_deltaf]
        fsk4_seq_n = random.randint(8, 16) # 码元宽度N/16 ~ N/8
        fsk4_seq = generate_nfsk_sequence(fsk4_seq_n, 4)
        _, fsk4_signal = generate_fsk_signal(sample_num=SAMPLE_NUM, sample_rate=SAMPLE_RATE, freq_list=fsk4_list,
                                            sequence=fsk4_seq)
        fsk4_signal = awgn(fsk4_signal, snr)
        fsk4_snr.append(fsk4_signal)
        # BPSK
        bpsk_f0 = random.uniform(SAMPLE_RATE * 0.05, SAMPLE_RATE * 0.45)
        bpsk_n = random.choice([2, 3, 4, 5, 7, 11, 13])
        bpsk_seq = bark_code(n=bpsk_n)
        _, bpsk_signal = generate_bpsk_signal(sample_num=SAMPLE_NUM, sample_rate=SAMPLE_RATE,
                                            sequence=bpsk_seq, f0=bpsk_f0)
        bpsk_signal = awgn(bpsk_signal, snr)
        bpsk_snr.append(bpsk_signal)
        # Frank
        frank_f0 = random.uniform(SAMPLE_RATE * 0.1, SAMPLE_RATE * 0.4)
        frank_P = random.randint(4, 7)
        _, frank_signal = generate_frank_signal(sample_num=SAMPLE_NUM, sample_rate=SAMPLE_RATE,
                                              f0=frank_f0, P=frank_P)
        frank_signal = awgn(frank_signal, snr)
        frank_snr.append(frank_signal)

        # LFM-BPSK
        # todo 参数需要标定
        lfm_bpsk_n = random.randint(4, 13)
        lfm_bpsk_seq = np.random.randint(0, 2, lfm_bpsk_n)
        lfm_bpsk_f0 = random.uniform(SAMPLE_RATE * 0.05, SAMPLE_RATE * 0.4) # 0.05~0.4
        lfm_bpsk_f1 = random.uniform(0.1 * SAMPLE_RATE, 0.5 * SAMPLE_RATE - lfm_bpsk_f0) + lfm_bpsk_f0 # 0.1~0.5, 带宽最小为0.1
        _, lfm_bpsk_signal = generate_lfm_bpsk_signal(sample_num=SAMPLE_NUM, sample_rate=SAMPLE_RATE, f0=lfm_bpsk_f0,
                                                      f1=lfm_bpsk_f1, sequence=lfm_bpsk_seq)
        lfm_bpsk_signal = awgn(lfm_bpsk_signal, snr)
        lfm_bpsk_snr.append(lfm_bpsk_signal)

    data[('LFM', snr)] = lfm_snr
    data[('CW', snr)] = cw_snr
    data[('VTFM', snr)] = vtfm_snr
    data[('SFM', snr)] = sfm_snr
    data[('EQFM', snr)] = eqfm_snr
    data[('2FSK', snr)] = fsk2_snr
    data[('4FSK', snr)] = fsk4_snr
    data[('BPSK', snr)] = bpsk_snr
    data[('FRANK', snr)] = frank_snr
    data[('LFM-BPSK', snr)] = lfm_bpsk_snr
    print("snr {} finish".format(snr))


with open('data/data_classifier_t.pkl', 'wb') as f:
    pickle.dump(data, f)
    f.close()
    print('data saved')

