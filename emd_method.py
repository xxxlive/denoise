import librosa
import numpy as np
import soundfile

import PyEMD.EMD as EMD
import matplotlib.pyplot as plt
import librosa.display
import savitzkygolay as sa


def to_overlap_frame(audio_arr, sr, overlap=0):
    res = []
    length = len(audio_arr)
    frame_time = 1 / 4
    frame_length = int(frame_time * sr)
    overlap = int(frame_length / 2)
    st = 0
    tem = []
    while True:
        if st + frame_length < length:
            tem = audio_arr[st:st + frame_length:1]
        else:
            # print('yes')
            tem = audio_arr[st:]
            res.append(np.array(tem))
            break
        st = st + overlap
        res.append(np.array(tem))
    return res, overlap


def overlap_join(audio_arrs, overlap):
    emd = EMD()
    res = [[]]
    print(len(audio_arrs))
    for audio_arr in audio_arrs:
        # print(len(audio_arr))
        imf_s = emd.emd(audio_arr, max_imf=10)
        tem = np.zeros(len(audio_arr))
        for i in range(1, len(imf_s)):
            tem = tem + imf_s[i]
        res.append(tem)
    ret = res[1]
    for i in range(3, len(res)):
        t1 = np.array(res[i])
        # t2 = np.array(res[i + 1])
        # tt = np.append(t1[:overlap], t1[overlap:overlap + min(len(t2), overlap)] + t2[:min(len(t2), overlap)])
        sum_len = min(overlap, len(t1))
        # tot = len(t1)
        print(sum_len)
        ret[len(ret) - sum_len:] = ret[len(ret) - sum_len:] + t1[:sum_len]
        ret = np.append(ret, t1[sum_len:])
    ret[:] /= 2
    ret[:overlap] *= 2
    ret[:-overlap] *= 2
    return ret


def to_frame(audio_arr, sr):
    frame_time = 1 / 10
    frame_length = int(frame_time * sr)
    res = np.array_split(audio_arr, frame_length)
    return res


def EMD_join(audio_arrs):
    res = np.array([])
    for audio_arr in audio_arrs:
        emd = EMD()
        imf_s = emd.emd(audio_arr)
        tem = np.zeros(len(audio_arr))
        for i in range(1, len(imf_s)):
            tem = tem + imf_s[i]
        res = np.append(res, tem)
    return res


audio_arr, sr = librosa.load("./noise10/english_noise_10.wav")
# times = np.linspace(0, len(audio_arr) / sr, len(audio_arr))
# audio_arrs, overlap = to_overlap_frame(audio_arr, sr)
# audio_denoise = overlap_join(audio_arrs, overlap)
# soundfile.write("english_denoise.wav", audio_denoise, sr)

# emd分解
plt.figure(1, figsize=(4, 20))
emd = EMD()
audio_IMFS = emd.emd(audio_arr, max_imf=10)
imfs_num = audio_IMFS.shape[0]
fig, ax = plt.subplots(nrows=imfs_num, sharex=True, figsize=(4, 20))
for i in range(imfs_num):
    librosa.display.waveshow(audio_IMFS[i], sr=sr, ax=ax[i], color='g')
    ax[i].set(title=('imf' + str(i + 1)))
    ax[i].label_outer()
    # plt.ylabel("IMF" + str(i + 1))
# plt.savefig('./noise0/emd_result.jpg')
plt.show()
# 去掉最高频的一条
sum_imf = np.zeros(len(audio_arr))
for i in range(1, imfs_num - 1):
    sum_imf = sum_imf + audio_IMFS[i]

plt.figure(2)
fig2, ax2 = plt.subplots(nrows=3, sharex=True)
librosa.display.waveshow(np.array(audio_arr), sr=sr, ax=ax2[0])
librosa.display.waveshow(np.array(sum_imf), sr=sr, ax=ax2[1])
librosa.display.waveshow(np.array(audio_arr), sr=sr, ax=ax2[2], color='r')
librosa.display.waveshow(np.array(sum_imf), sr=sr, ax=ax2[2], color='g')

soundfile.write("./noise10/english_emd_del1.wav", sum_imf, sr)

plt.show()
