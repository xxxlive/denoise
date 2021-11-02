import librosa
import librosa.display
import IPython.display as ipd
import numpy
import soundfile
from skimage.restoration import denoise_wavelet
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import use_func
import sgolay

root = "./noise10/"

fig, ax = plt.subplots(nrows=3, sharex=True)
audio_arr, sr = librosa.load(root + "/english_noise_10.wav", sr=None)
# audio_arr_denoise = denoise_wavelet(audio_arr, wavelet_levels=3)
audio_arr_denoise = use_func.movmean(audio_arr, 5)
# audio_arr_denoise = sgolay.savgol(audio_arr, 5, 4)
times = numpy.linspace(0, len(audio_arr) / sr, len(audio_arr))
# plt.subplot(2, 1, 1)
# plt.subplot(2, 1, 2)
librosa.display.waveshow(audio_arr, sr, ax=ax[0])
librosa.display.waveshow(audio_arr_denoise, sr, ax=ax[1])
librosa.display.waveshow(audio_arr, sr, ax=ax[2])
librosa.display.waveshow(audio_arr_denoise, sr, ax=ax[2])
plt.show()
soundfile.write(root + "english_denoise_movemean.wav", audio_arr, sr)
