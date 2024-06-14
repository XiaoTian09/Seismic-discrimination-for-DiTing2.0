import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import obspy
import pandas as pd


# 将加州数据设置为z分量

# load json dictionary
DiTing_2021_2022_json_file = '../../DiTing_2021_2022.json'
DiTing_2021_2022_json = json.load(open(DiTing_2021_2022_json_file, 'r'))
DiTing_2021_2022_HDF_file = '../../DiTing_2021_2022.hdf5'
DiTing_2021_2022_HDF = h5py.File(DiTing_2021_2022_HDF_file, 'r')

#查看hdf5文件中的键值
print("hdf5 key:",list(DiTing_2021_2022_HDF.keys()))
key_list = list(DiTing_2021_2022_json.keys())#将json中的键值放入列表中

#将key_list的总数算出来，之后使用for函数对数据进行循环计算
print('总数为:',len(key_list))



natural_data=[]
for key in key_list[:8000]:#将地震类型符合的参数加入keys中,这里设置key值为10000
    data_from_json = DiTing_2021_2022_json[key]
    # keys.append(key)
    # get the data from the HDF file
    data = DiTing_2021_2022_HDF.get('earthquake').get(key)[()]#读取波形数据 [()]
    data=data[:,0]
    # print('从中获取其中数据的基本信息:',data_from_json)
    # print('数据的键值为:',data_from_json.keys())
    #对数据去空值
    data = np.array(data)
    # data = data_numpy[~np.isnan(data_numpy)]
    data = data[~np.isnan(data)]
    # 对数据去趋势
    data_detrended = signal.detrend(data, axis=0, type='linear')
    data_len = len(data_detrended)
    window = signal.windows.hann(data_len)
    # 对数据进行平滑
    data_detrended_smoothed = signal.convolve(data_detrended, window, mode='same') / sum(window)  # 平滑
    data_detrended = data_detrended - data_detrended_smoothed
    # 对数据进行标准化
    std = np.std(data_detrended)
    data_normalized = data_detrended / std
    # 带通滤波
    Fs = 50  # 采样率
    c, d = signal.butter(4, [1 / (Fs / 2), 15 / (Fs / 2)], 'bandpass')  # 滤波器系数
    data_filtered = signal.filtfilt(c, d, data_normalized, axis=0)
    if np.isnan(data_filtered).any():
        continue
    #数据切割
    ref_time = obspy.UTCDateTime(key.split('_')[0])#对数据时间进行分割
    if 'Pg' in data_from_json.keys():
        t_time =data_from_json['Pg']
        print('p波到达时间：', t_time)
        p_arrival_time = x=(obspy.UTCDateTime(t_time) - ref_time + 30.0) * 100
        p_arrival_time=int(p_arrival_time)
        # 截取从 P 波到达前 1000 开始到第 3000 个样点
        print(p_arrival_time)
        data_filtered = data_filtered[p_arrival_time - 1000: p_arrival_time + 7000]
        # plt.plot(data_filtered)
        target_length = 8000
        if len(data_filtered) < target_length:
            pad_width = ((0, target_length - len(data_filtered)),)  # 指定每个维度需要填充的长度
            data_filtered = np.pad(data_filtered, pad_width=pad_width, mode="constant", constant_values=0)  # 填充 0
        natural_data.append(data_filtered)
    else:
        pass
        # t_time = data_from_json['Pn']
        # print('p波到达时间：', t_time)
        # p_arrival_time = x = (obspy.UTCDateTime(t_time) - ref_time + 30.0) * 100
        # p_arrival_time = int(p_arrival_time)
        # # 截取从 P 波到达前 1000 开始到第 3000 个样点
        # print(p_arrival_time)
        # data_filtered = data_filtered[p_arrival_time - 1000: p_arrival_time + 3000]
        # # plt.plot(data_filtered)
        # target_length = 4000
        # if len(data_filtered) < target_length:
        #     pad_width = ((0, target_length - len(data_filtered)),)  # 指定每个维度需要填充的长度
        #     data_filtered = np.pad(data_filtered, pad_width=pad_width, mode="constant", constant_values=0)  # 填充 0
        # natural_data.append(data_filtered)



plt.plot(natural_data[1020])
plt.show()

natural_datas=np.array(natural_data)
natural_datas = np.apply_along_axis(lambda x: (x-min(x))/(max(x)-min(x)), axis=1, arr=natural_datas)
# plt.imshow(natural_datas)
print(len(natural_datas))
np.save('./data/natural_datas.npy',natural_datas)
