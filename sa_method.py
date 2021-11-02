import librosa
import librosa.display
import numpy as np
import soundfile
from matplotlib import pyplot as plt
import savitzkygolay.savitzkygolay as sa
import sgolay

root = './noise10/'
fig, ax = plt.subplots(nrows=3, sharex=True)
audio_arr, sr = librosa.load(root + "english_noise_10.wav")
# audio_arr = audio_arr[:1000]
# times = np.linspace(0, len(audio_arr) / sr, len(audio_arr))
# plt.plot(times, audio_arr, 'r')
audio_arr_denoise = sa.filter1D(audio_arr, 5, 3)
# audio_arr_denoise = sgolay.savgol(audio_arr, 7, 5)
# plt.plot(times, audio_arr_denoise[3:-3], 'g')
librosa.display.waveshow(audio_arr, sr, ax=ax[0])
librosa.display.waveshow(audio_arr_denoise, sr, ax=ax[1])
librosa.display.waveshow(audio_arr, sr, ax=ax[2])
librosa.display.waveshow(audio_arr_denoise, sr, ax=ax[2])
plt.show()
soundfile.write(root + "english_denoise_sa.wav", audio_arr, sr)
