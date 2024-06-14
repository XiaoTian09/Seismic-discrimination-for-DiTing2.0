# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 16:36:57 2024

@author: yangzj
"""

import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import obspy
from scipy import signal
from obspy.signal.trigger import classic_sta_lta

# load json dictionary
DiTing_non_natural_json_file = '../../DiTing_non_natural.json'
DiTing_non_natural_json = json.load(open(DiTing_non_natural_json_file, 'r'))

DiTing_non_natural_HDF_file = '../../DiTing_non_natural.hdf5'
DiTing_non_natural_HDF = h5py.File(DiTing_non_natural_HDF_file, 'r')

# 查看hdf5文件中的键值
print("hdf5 key:", list(DiTing_non_natural_HDF.keys()))
key_list = list(DiTing_non_natural_json.keys())  # 将json中的键值放入列表中

#将key_list的总数算出来，之后使用for函数对数据进行循环计算
print('总数为:',len(key_list))

#获取key值
keys=[]
for key in key_list:#将地震类型符合的参数加入keys中
    data_from_json = DiTing_non_natural_json[key]
    type =data_from_json['evtype']
    #print(type)
    if type == 'ep':#判断是不是爆破地震
        keys.append(key)
    else:
        pass
  
def STA_LTA(waveform_data,key):
    # 确定采样率
    sampling_rate = 100  # Hz
    
    # 将采样率转换为毫秒单位的时间间隔
    #dt = 1 / sampling_rate * 1000  # 毫秒
    
    # 定义STA与LTA窗口长度（以样本计）
    nsta = int(0.5 * sampling_rate)  # STA窗口长度可以为0.5秒作为示例
    nlta = int(2.0 * sampling_rate)  # LTA窗口长度可以为2秒作为示例
    
    # 计算STA/LTA比值
    sta_lta_values = classic_sta_lta(waveform_data, nsta=nsta, nlta=nlta)
    
    # 设定阈值，这通常需要根据信号特征和噪声水平来设定
    threshold = 2.8 # 举例，可能需要调整以适应实际数据
    
    # 寻找超过阈值的STA/LTA比值的位置
    
    detections = np.where(sta_lta_values > threshold)[0]
    
    # P波最可能的位置是STA/LTA峰值所在的时间戳
    if len(detections)>0:
        p_wave_indices = detections[0]  # 取第一个显著峰值作为P波的估计位置
    else:
        ref_time = obspy.UTCDateTime(key.split('_')[0])
        data_from_json = DiTing_non_natural_json[key]  #提取其中的json值
        t_time =data_from_json['Pg']
        print('p波到达时间：', t_time)
        p_wave_indices = x=(obspy.UTCDateTime(t_time) - ref_time + 30.0) * 100
    
    print("P_Wave：",p_wave_indices)
    return p_wave_indices



non_natural_data=[]
for key in keys:
    data = DiTing_non_natural_HDF.get('non_natural').get(key)[()]  # 读取波形数据 [()]
    
    #plt.figure()
    #plt.subplot(3,1,1)
    #plt.plot(data[:,0])
    
    
    data = data[:,0]
    #对数据去空值
    data_numpy = np.array(data)
    data = data_numpy[~np.isnan(data_numpy)]
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
    Fs = 50  # 采样率 #100最好
    c, d = signal.butter(4, [1 / (Fs / 2), 15 / (Fs / 2)], 'bandpass')  # 滤波器系数
    data_filtered = signal.filtfilt(c, d, data_normalized, axis=0)
    #plt.subplot(3,1,2)
    #plt.plot(data_filtered)
    

    if 'Pg' in data_from_json.keys():
        #p_arrival_time = STA_LTA(data_filtered,key)
        
        #p_arrival_time = int(p_arrival_time)
        p_arrival_time = 4530
        
        data_filtered = data_filtered[p_arrival_time - 1000: p_arrival_time + 7000]
        target_length = 8000
        if len(data_filtered) < target_length:
            pad_width = ((0, target_length - len(data_filtered)),)  # 指定每个维度需要填充的长度
            data_filtered = np.pad(data_filtered, pad_width=pad_width, mode="constant", constant_values=0)  # 填充 0
        non_natural_data.append(data_filtered)
        #plt.subplot(3,1,3)
        #plt.plot(data_filtered)
    else:
        pass

plt.plot(non_natural_data[1300])
plt.show()

non_natural_datas=np.array(non_natural_data)
non_natural_datas_norm = np.apply_along_axis(lambda x: (x-min(x))/(max(x)-min(x)), axis=1, arr=non_natural_datas)
# plt.imshow(non_natural_datas_norm)
np.save('./data/non_natural_datas_blasting.npy',non_natural_datas_norm)
