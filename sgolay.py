# -*- coding: utf-8 -*-
import numpy as np

"""
* 创建系数矩阵X
* size - 2×size+1 = window_size
* rank - 拟合多项式阶次
* x - 创建的系数矩阵
"""


def create_x(size, rank):
    x = []
    for i in range(2 * size + 1):
        m = i - size
        row = [m ** j for j in range(rank)]
        x.append(row)
    x = np.mat(x)
    return x


"""
 * Savitzky-Golay平滑滤波函数
 * data - list格式的1×n纬数据
 * window_size - 拟合的窗口大小
 * rank - 拟合多项式阶次
 * ndata - 修正后的值
"""


def savgol(data, window_size, rank):
    m = (window_size - 1) / 2
    odata = data[:]
    # 处理边缘数据，首尾增加m个首尾项
    for i in range(int(m)):
        np.insert(odata, 0, odata[0])
        # odata.insert(0, odata[0])
        # odata.insert(len(odata), odata[len(odata) - 1])
        np.insert(odata, len(odata), odata[len(odata) - 1])
    # 创建X矩阵
    x = create_x(m, rank)
    # 计算加权系数矩阵B
    b = (x * (x.T * x).I) * x.T
    a0 = b[m]
    a0 = a0.T
    # 计算平滑修正后的值
    ndata = []
    for i in range(len(data)):
        y = [odata[i + j] for j in range(window_size)]
        y1 = np.mat(y) * a0
        y1 = float(y1)
        ndata.append(y1)
    return ndata
