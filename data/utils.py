import numpy as np
import scipy.signal as signal
from scipy.ndimage import zoom

def awgn(x, snr, out='signal', method='vectorized', axis=0):
    """
    向信号x添加指定信噪比的高斯白噪声
    :param x:
    :param snr: dB
    :param out:
    :param method:
    :param axis:
    :return:
    """
    # Signal power
    if method == 'vectorized':
        N = x.size
        Ps = np.sum(x ** 2 / N)
    elif method == 'max_en':
        N = x.shape[axis]
        Ps = np.max(np.sum(x ** 2 / N, axis=axis))
    elif method == 'axial':
        N = x.shape[axis]
        Ps = np.sum(x ** 2 / N, axis=axis)
    else:
        raise ValueError('method \"' + str(method) + '\" not recognized.')
    # Signal power, in dB
    Psdb = 10 * np.log10(Ps)
    # Noise level necessary
    Pn = Psdb - snr
    # Noise vector (or matrix)
    n = np.sqrt(10 ** (Pn / 10)) * np.random.normal(0, 1, x.shape)
    if out == 'signal':
        return x + n
    elif out == 'noise':
        return n
    elif out == 'both':
        return x + n, n
    else:
        return x + n


def bark_code(n: int) -> list:
    """
    巴克码
    :param n: 巴克码长度
    :return:
    """
    if n == 2:
        return [1, 0]
    if n == 3:
        return [1, 1, 0]
    if n == 4:
        return [1, 1, 0, 1]
    if n == 5:
        return [1, 1, 1, 0, 1]
    if n == 7:
        return [1, 1, 1, 0, 0, 1, 0]
    if n == 11:
        return [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]
    if n == 13:
        return [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
    raise ValueError('n must be 2, 3, 4, 5, 7, 11, 13')


def preprocessing(x: np.ndarray, window:str = 'hann', nperseg:int = 64, noverlap:int = 32, target_shape:tuple = (40, 40)) -> np.ndarray:
    """
    信号预处理，做短时傅里叶变换，取能量谱对数，归一化，缩放40*40
    :param x: 信号
    :param window: FFT窗函数
    :param nperseg: 窗大小
    :param noverlap: 重叠大小
    :param target_shape: 输出大小
    :return: 归一化时频谱(40x40)
    """
    f, ts, Zxx = signal.stft(x, 1, window=window, nperseg=nperseg, noverlap=noverlap)
    Zxx_abs = np.abs(Zxx)
    # print(Zxx_abs.shape)
    Zxx_abs = (Zxx_abs - np.min(Zxx_abs))/(np.max(Zxx_abs) - np.min(Zxx_abs))
    Zxx_abs = zoom(Zxx_abs, (target_shape[0]/Zxx_abs.shape[0], target_shape[1]/Zxx_abs.shape[1]), order=1)
    return Zxx_abs


def generate_nfsk_sequence(seq_length: int, n_fsk: int) -> np.ndarray:
    """生成NFSK序列"""
    base_code = np.arange(n_fsk)
    remain_len = seq_length - n_fsk
    remain_code = np.random.randint(0, n_fsk, size=remain_len)
    full_seq = np.concatenate((base_code, remain_code))
    np.random.shuffle(full_seq)
    return full_seq




if __name__ == '__main__':
    print(generate_nfsk_sequence(5, 4))