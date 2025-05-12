import numpy as np
from numpy import ndarray


def generate_cw_signal(duration: float, sample_rate: float,
                       f0: float, amplitude: float = 1.0, phi0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    生成CW信号（单边波）
    :param duration: 信号持续时间（秒）
    :param sample_rate: 采样率（Hz）
    :param f0: 信号频率（Hz）
    :param amplitude: 信号幅度
    :param phi0: 信号初始相位
    :return: 时间数组和CW信号
    """
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    cw_signal = amplitude * np.cos(2 * np.pi * f0 * t)
    return t, cw_signal


def generate_lfm_signal(duration: float, sample_rate: float,
                        f0: float, f1: float, amplitude: float = 1.0, phi0:float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    生成LFM信号（啁啾信号）
    :param duration: 信号持续时间（秒）
    :param sample_rate: 采样率（Hz）
    :param f0: 起始频率（Hz）
    :param f1: 结束频率（Hz）
    :param amplitude: 信号幅度
    :param phi0: 初始相位
    :return: 时间数组和LFM信号
    """
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    k = (f1 - f0) / duration  # 频率变化率, B = f1 - f0.
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t ** 2) + phi0
    lfm_signal = amplitude * np.cos(phase)
    return t, lfm_signal


def generate_vfm_signal(duration: float, sample_rate: float,
                        f0: float, B: float, amplitude: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    产生V型调频信号
    s(t) = A*cos(2*pi*f0*t + pi*d1(t)*t**2)
             k if 0 < t < T/2
    d1(t) =
             -k + kT/t if T/2 < t < T
    k = B/(T/2)
    :param duration: 脉冲持续时间
    :param sample_rate: 采样率
    :param f0: 起始频率
    :param B: 带宽
    :param amplitude: 幅值
    :return: 时间数组和VFM信号
    """
    # TODO 为什么这个实现不正确
    # t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    # k = B / (duration / 2)
    # d1 = np.zeros_like(t)
    # d1[t < duration / 2] = k
    # d1[t >= duration / 2] = -k + k*duration / t[t >= duration / 2]
    # phase = 2 * np.pi * f0 * t + np.pi * d1 * t**2
    # vfm_signal = amplitude * np.cos(phase)
    # return t, vfm_signal

    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    k = B / (duration / 2)
    # 瞬时频率 f(t)
    f_t = np.piecewise(
        t,
        [t < duration / 2, t >= duration / 2],
        [lambda ts: f0 + k * ts, lambda ts: f0 + k * (duration - ts)]
    )
    # 瞬时相位 phi(t)，通过对频率积分得到
    phase = 2 * np.pi * np.cumsum(f_t) / sample_rate
    vfm_signal = amplitude * np.cos(phase)
    return t, vfm_signal


def generate_sfm_signal(duration: float, sample_rate: float,
                        f0: float, B: float, mo: float, amplitude=1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    产生正弦调制信号
    s(t) = A*cos(2*pi*f0*t + mf*sin(2*pi*fmo*t))
    fmo = mo/duration
    mf = B/fmo
    :param duration: 信号持续时间
    :param sample_rate: 采样率
    :param f0: 起始频率
    :param B: 带宽
    :param mo: 正弦调频周期参数
    :param amplitude: 幅值
    :return: 时间数组和SFM信号
    """
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    fmo = mo / duration
    mf = B / 2 / fmo
    phase = 2 * np.pi * f0 * t + mf * np.sin(2 * np.pi * fmo * t)
    sfm_signal = amplitude * np.cos(phase)
    return t, sfm_signal


def generate_eqfm_signal(duration: float, sample_rate: float,
                         f0: float, B: float, amplitude: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    产生偶二次调频信号
    s(t) = A*cos(2*pi*(f0*t + k/3*(t - T/2)**3))
    k = 4*B/T**2
    :param duration: 脉冲持续时间
    :param sample_rate: 采样率
    :param f0: 起始频率
    :param B: 带宽
    :param amplitude: 幅值
    :return: 时间数组和EQFM信号
    """
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    k = 4 * B / (duration ** 2)
    freq = f0 + k * (t - duration / 2) ** 2
    phrase = 2 * np.pi * np.cumsum(freq) / sample_rate
    eqfm_signal = amplitude * np.cos(phrase)
    return t, eqfm_signal


def generate_fsk_signal(sample_num: int, sample_rate: float,
                        freq_list: list, sequence: list | ndarray, amplitude: float = 1.0, phi0: float = 0.0) -> tuple[
    np.ndarray, np.ndarray]:
    """
    产生NFSK信号，不足补0
    :param sample_num: 信号采样点数
    :param sample_rate: 采样率
    :param freq_list: 信号频率列表, 长度代表NFSK
    :param sequence: 信号序列
    :param amplitude:  信号幅度
    :param phi0: 初相
    :return: 时间数组和FSK信号
    """
    n = len(freq_list)
    if any(bit not in range(n) for bit in sequence):
        raise ValueError("except value in datalist, expect range({}).".format(list(range(n))))
    TbN = sample_num // len(sequence)  # 码元宽度
    t = np.linspace(0, sample_num / sample_rate, sample_num, endpoint=False)
    fsk_signal = np.zeros_like(t)

    current_phase = phi0
    # 生成2FSK信号
    for i, bit in enumerate(sequence):
        # print("i:{}, bit:{}".format(i, bit))
        # 当前码元的时间范围
        start_idx = i * TbN
        end_idx = (i + 1) * TbN
        # 根据当前比特选择频率
        freq = freq_list[bit]
        # 保证相位连续
        delta_phase = 2 * np.pi * freq * (t[start_idx:end_idx] - t[start_idx])
        phase = current_phase + delta_phase
        # 生成当前码元的信号
        fsk_signal[start_idx:end_idx] = amplitude * np.cos(phase)
        current_phase += 2 * np.pi * freq * (t[end_idx - 1] - t[start_idx])  # 更新当前相位
        # print("current_phase:{}".format(current_phase))
    return t, fsk_signal


def generate_bpsk_signal(sample_num: int, sample_rate: float,
                         sequence: list | ndarray, f0: float, amplitude: float = 1.0, phi0: float = 0.0) -> tuple[
    np.ndarray, np.ndarray]:
    """
    产生BPSK信号, 不足补0
    :param sample_num: 信号采样点数
    :param sample_rate: 采样率
    :param sequence: 信号序列
    :param f0: BPSK载频
    :param amplitude: 信号幅度
    :param phi0: 初相
    :return: 时间数组和BPSK信号
    """
    if any(bit not in [0, 1] for bit in sequence):
        raise ValueError("except value in datalist, expect range({}).".format([0, 1]))

    t = np.linspace(0, sample_num / sample_rate, sample_num, endpoint=False)
    bpsk_signal = np.zeros_like(t)
    TbN = sample_num / len(sequence)  # 码元宽度
    for i, bit in enumerate(sequence):
        start_idx = i * int(TbN)
        end_idx = (i + 1) * int(TbN)
        _, sub_signal = generate_cw_signal(TbN/sample_rate, sample_rate, f0, amplitude, 0)
        if bit == 1:  # 为1跳变
            _, sub_signal = generate_cw_signal(TbN/sample_rate, sample_rate, f0, amplitude, np.pi)
        bpsk_signal[start_idx:end_idx] = sub_signal
    return t, bpsk_signal


def generate_frank_signal(sample_num: int, sample_rate: float,
                          f0: float, P: int, amplitude: float = 1.0, phi0: float = 0.0):
    """
    生成Frank编码的多相调制信号
    :param sample_num: 总采样点数
    :param sample_rate: 采样率 (Hz)
    :param f0: 载频 (Hz)
    :param P: Frank编码的阶数
    :param amplitude: 幅值
    :param phi0: 初相 (弧度)
    :return: 时间数组和frank信号
    """
    # 计算时间数组
    # t = np.linspace(0, sample_num / sample_rate, sample_num)

    # 每个子脉冲的采样点数
    sub_pulse_samples = sample_num // P ** 2

    # 如果总采样点数不能被 P^2 整除，其后补0
    t = np.linspace(0, sample_num / sample_rate, sample_num, endpoint=False)

    # 生成Frank编码相位矩阵
    phase_matrix = np.zeros((P, P))
    for m in range(P):
        for n in range(P):
            phase_matrix[m, n] = (2 * np.pi / P) * m * n

    # 按行展开为一维相位序列
    phase_sequence = phase_matrix.ravel()

    # 初始化信号
    signal = np.zeros(sample_num)

    # 生成信号
    for i in range(P ** 2):
        # 当前子脉冲的起始和结束索引
        start_idx = i * sub_pulse_samples
        end_idx = (i + 1) * sub_pulse_samples

        # 当前子脉冲的相位
        current_phase = phase_sequence[i]

        # 生成当前子脉冲的正弦波
        signal[start_idx:end_idx] = amplitude * np.cos(2 * np.pi * f0 * t[start_idx:end_idx] + current_phase + phi0)

    return t, signal


# def generate_lfm_bpsk_signal(sample_num: int, sample_rate: float,
#                              f0: float, f1: float, sequence: list | ndarray, amplitude: float = 1.0,
#                              phi0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
#     """
#     生BPSK-LFM混合调制信号
#     :param sample_num: 采样点数
#     :param sample_rate: 采样率
#     :param f0: LFM起始频率
#     :param f1: LFM终止频率
#     :param sequence: BPSK序列
#     :param amplitude: 幅度
#     :param phi0: 初始相位
#     :return: 时间数组和lfm-bpsk信号
#     """
#     if any(bit not in [0, 1] for bit in sequence):
#         raise ValueError("except value in datalist, expect range({}).".format([0, 1]))
#
#     t = np.linspace(0, sample_num/sample_rate, sample_num, endpoint=False)
#     sub_pulse_samples = sample_num/len(sequence) # 子脉冲长度
#     sub_pulse_duration = sub_pulse_samples/sample_rate # 子脉冲持续时间
#     signal = np.zeros(sample_num)
#
#     for i, bit in enumerate(sequence):
#         start_idx = i * int(sub_pulse_samples) # 子脉冲起始
#         end_idx = (i + 1) * int(sub_pulse_samples) # 终止
#         _, sub_pulse_signal = generate_lfm_signal(sub_pulse_duration, sample_rate, f0, f1, amplitude, phi0)
#         if bit == 1:
#             _, sub_pulse_signal = generate_lfm_signal(sub_pulse_duration, sample_rate, f0, f1, amplitude, phi0 + np.pi)
#         signal[start_idx:end_idx] = sub_pulse_signal
#
#     return t, signal


def generate_lfm_bpsk_signal(sample_num: int, sample_rate: float,
                             f0: float, f1: float, sequence: list | ndarray, amplitude: float = 1.0,
                             phi0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    生BPSK-LFM混合调制信号
    :param sample_num: 采样点数
    :param sample_rate: 采样率
    :param f0: LFM起始频率
    :param f1: LFM终止频率
    :param sequence: BPSK序列
    :param amplitude: 幅度
    :param phi0: 初始相位
    :return: 时间数组和lfm-bpsk信号
    """
    if any(bit not in [0, 1] for bit in sequence):
        raise ValueError("except value in datalist, expect range({}).".format([0, 1]))
    duration = sample_num/sample_rate
    t = np.linspace(0, sample_num/sample_rate, sample_num, endpoint=False)
    sub_pulse_samples = sample_num/len(sequence) # 子脉冲长度
    sub_pulse_duration = sub_pulse_samples/sample_rate # 子脉冲持续时间
    delta_f = sub_pulse_duration/duration * (f1 - f0)
    signal = np.zeros(sample_num)

    for i, bit in enumerate(sequence):
        start_idx = i * int(sub_pulse_samples) # 子脉冲起始
        end_idx = (i + 1) * int(sub_pulse_samples) # 终止
        _, sub_pulse_signal = generate_lfm_signal(sub_pulse_duration, sample_rate,
                                                  f0 + (i - 1)*delta_f, f0 + i*delta_f, amplitude, phi0)
        if bit == 1:
            _, sub_pulse_signal = generate_lfm_signal(sub_pulse_duration, sample_rate,
                                                      f0 + (i - 1)*delta_f, f0 + i*delta_f, amplitude, phi0 + np.pi)
        signal[start_idx:end_idx] = sub_pulse_signal

    return t, signal


def generate_fsk_bpsk_signal():
    raise NotImplementedError






