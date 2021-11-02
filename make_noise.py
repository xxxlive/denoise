import librosa
import librosa.display
import numpy as np
import wave
import soundfile
from matplotlib import pyplot as plt
import compare

import numpy as np


def gen_gaussian_noise(signal, SNR):
    """
    :param signal: 原始信号
    :param SNR: 添加噪声的信噪比
    :return: 生成的噪声
    """
    noise = np.random.randn(*signal.shape)  # *signal.shape 获取样本序列的尺寸
    noise = noise - np.mean(noise)
    signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))
    noise_variance = signal_power / np.power(10, (SNR / 10))
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
    return noise


def awgn(x, snr, out='signal', method='vectorized', axis=0):
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


audio_arr, sr = librosa.load("english.wav")
# plt.figure(fi)
# plt.subplots_adjust(top=1)
fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6, 6))

librosa.display.waveshow(audio_arr, sr=sr, ax=ax[0])
snr = 10
max_wave = abs(max(audio_arr))
audio_arr = audio_arr / max_wave
# audio_noise_arr = gen_gaussian_noise(audio_arr, snr)
# audio_noise_arr = audio_noise_arr + audio_arr
audio_noise_arr = awgn(audio_arr, snr)
audio_noise_arr = audio_noise_arr * max_wave
librosa.display.waveshow(audio_noise_arr, sr=sr, ax=ax[1])
ax[0].set(title='clean')
# ax[0].label_outer()

librosa.display.waveshow(audio_noise_arr, color='r', sr=sr, ax=ax[2])
librosa.display.waveshow(audio_arr, color='g', sr=sr, ax=ax[2])
ax[1].set(title='with_noise')
ax[2].set(title='mixed')
plt.tight_layout()
plt.show()
soundfile.write("./noise10/english_noise_10.wav", audio_noise_arr, sr)
# print(compare.check_snr(audio_arr, audio_noise_arr - audio_arr))
print('信噪比检测结果为' + str(compare.check_snr_file('english.wav', './noise10/english_noise_10.wav')))
