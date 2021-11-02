import librosa
import numpy as np
import soundfile
import matplotlib.pyplot as plt
from ThinkDSP.code import thinkdsp, thinkplot
import librosa.display


def keep_percent_alive(percent, T, fre):
    r = 2000
    l = 1
    cnt = 0
    indi = []

    while l < r:
        mid = (l + r) / 2
        indi = T > mid
        print("averge    {},mid {}".format(sum(indi) / len(T), mid))
        tem, tot = 0, 0
        for i in range(len(T)):
            if 100 < fre[i] < 1000 and indi[i]:
                tem += 1
            else:
                indi[i] = False
            if 100 < fre[i] < 1000:
                tot += 1

        if (tem / tot) >= percent:
            break
        else:
            r /= 2
        cnt += 1
        if cnt > 1000:
            break
    return indi


root = './noise10/'

'''
直接使用fft，自己实现
'''
fig, ax = plt.subplots(nrows=3, sharex=True)
audio_arr, sr = librosa.load(root + '/english_noise_10.wav')
librosa.display.waveplot(audio_arr, sr=sr, color='r', ax=ax[0])
audio_arr_fft = np.fft.fft(audio_arr, len(audio_arr))
dt = 1 / sr
n = len(audio_arr)
PSD = audio_arr_fft * np.conj(audio_arr_fft)  # power spectrum
freq = (1 / (dt * n)) * np.arange(n)
indicies = keep_percent_alive(0.8, PSD, freq)
PSD_clean = PSD * indicies
audio_arr_fft = audio_arr_fft * indicies
ifft = np.fft.ifft(audio_arr_fft)
ifft = np.real(ifft)
librosa.display.waveplot(ifft, sr=sr, color='g', ax=ax[1])
librosa.display.waveplot(audio_arr, sr=sr, color='r', ax=ax[2])
librosa.display.waveplot(ifft, sr=sr, color='g', ax=ax[2])
plt.show()
soundfile.write(root + 'english_denoise_fft.wav', ifft, sr)

'''
使用thinkdsp
'''

# thinkplot.preplot(3, 1)
# audio_wave = thinkdsp.read_wave(root + 'english_noise_10.wav')
# thinkplot.subplot(1, 3, 1)
# audio_wave.plot(color='r')
# audio_spectrum = audio_wave.make_spectrum()
# audio_spectrum.band_stop(low_cutoff=100, high_cutoff=1000, factor=0.01)
# audio_wave_denoise = audio_spectrum.make_wave()
# thinkplot.subplot(2, rows=3, cols=1)
# audio_wave_denoise.plot(color='g')
# thinkplot.subplot(3, 3, 1)
# audio_wave.plot()
# audio_wave_denoise.plot()
# thinkplot.config(xlabel='Time')
# plt.show()
# audio_wave_denoise.write(root + 'english_denoise_fft_thinkdsp.wav')
