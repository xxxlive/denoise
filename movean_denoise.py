import librosa
import numpy as np
import soundfile
import use_func as fuc
import matplotlib.pyplot as plt

audio_arr, sr = librosa.load('english.wav')
times = np.linspace(0, (len(audio_arr) / sr), (len(audio_arr)))
audio_arr_denoise = fuc.movmean(audio_arr, 5)
soundfile.write('english_denoise.wav', audio_arr, sr)
# plt.subplot(2, 1, 1)
plt.plot(times, audio_arr, 'r')
# plt.subplot(2, 1, 2)
plt.plot(times, audio_arr_denoise, 'g')
plt.show()
