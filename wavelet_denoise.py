import librosa
import librosa.display
import numpy as np
import soundfile
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt

root = './noise10/'
fig, ax = plt.subplots(nrows=3, sharex=True)
audio_arr, sr = librosa.load(root + 'english_noise_10.wav')
audio_arr_denoise = denoise_wavelet(audio_arr, wavelet='db1', wavelet_levels=4, rescale_sigma=True)
librosa.display.waveshow(audio_arr, sr, ax=ax[0])
librosa.display.waveshow(audio_arr_denoise, sr, ax=ax[1])
librosa.display.waveshow(audio_arr, sr, ax=ax[2])
librosa.display.waveshow(audio_arr_denoise, sr, ax=ax[2])
plt.show()
soundfile.write(root + "english_denoise_wavelet.wav", audio_arr, sr)
