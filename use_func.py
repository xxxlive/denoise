import numpy as np


def movmean(T, m):
    assert (m <= T.shape[0])
    n = T.shape[0]
    sums = np.zeros(n - m + 1)
    sums[0] = np.sum(T[0:m])
    cumsum = np.cumsum(T)
    cumsum = np.insert(cumsum, 0, 0)  # 在数组开头插入一个0
    sums = cumsum[m:] - cumsum[:-m]
    sums = np.append(np.array(T[:m-1]), sums)
    return sums / m
