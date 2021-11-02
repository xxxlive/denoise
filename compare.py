import numpy as np
import librosa
import pandas as pd
from typing import List


def check_snr(signal, noise):
    """
    :param signal: 原始信号
    :param noise: 生成的高斯噪声
    :return: 返回两者的信噪比
    """
    signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))  # 0.5722037
    noise_power = (1 / noise.shape[0]) * np.sum(np.power(noise, 2))  # 0.90688
    SNR = 10 * np.log10(signal_power / noise_power)
    return SNR


def check_snr_file(clean_file, original_file):
    clean_audio, sr = librosa.load(clean_file)
    original_audio, sr = librosa.load(original_file)
    length = min(len(clean_audio), len(original_audio))
    noise_audio = (original_audio[:length]) - (clean_audio[:length])
    return check_snr(clean_audio, noise_audio)


# 计算信噪比
def SNR_singlech(clean_file, original_file):
    clean, clean_fs = librosa.load(clean_file, sr=None, mono=True)  # 导入干净语音
    ori, ori_fs = librosa.load(original_file, sr=None, mono=True)  # 导入原始语音
    length = min(len(clean), len(ori))
    est_noise = ori[:length] - clean[:length]  # 计算噪声语音

    # 计算信噪比
    SNR = 10 * np.log10((np.sum(clean ** 2)) / (np.sum(est_noise ** 2)))
    return SNR


root = './noise10/'
fft_rate = check_snr_file('english.wav', root + 'english_denoise_fft.wav')
fft_thinkdsp_rate = check_snr_file('english.wav', root + 'english_denoise_fft_thinkdsp.wav')
movemean_rate = check_snr_file('english.wav', root + 'english_denoise_movemean.wav')
wavelet_rate = check_snr_file('english.wav', root + 'english_denoise_wavelet.wav')
emd1_rate = check_snr_file('english.wav', root + 'english_emd_del1.wav')
emd2_rate = check_snr_file('english.wav', root + 'english_emd_del2.wav')
sa_rate = check_snr_file('english.wav', root + 'english_denoise_sa.wav')
result = [fft_rate, fft_thinkdsp_rate, movemean_rate, wavelet_rate, emd1_rate, emd2_rate, sa_rate]
frame = pd.DataFrame([result], columns=['fft', 'fft_thinkdsp', 'movemean', 'wavelet', 'emd1', 'emd2', 'savitzkygolay'])
frame.to_csv(path_or_buf=root + 'result.csv', index=False)
