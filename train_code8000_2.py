# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 10:26:59 2023

@author: yangzj
"""
import math
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Layer,GlobalAveragePooling1D,ReLU

from tensorflow.keras import optimizers
from tensorflow.keras.layers import *
import numpy as np
import os
from keras import callbacks
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, save_model, load_model
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
time_start = time.time()
import keras
from keras.utils import np_utils
import tensorflow as tf

# 加载数据集

# data1 = np.load('natural_datas.npy')
# labels1 = np.load('nature_ear_label.npy')
#
# data2 = np.load('non_natural_datas.npy')
# labels2 = np.load('non_nature_ear_label.npy')
#
#
#
# X = np.concatenate((data1, data2), axis=0)#加州数据
# y = np.concatenate((labels1, labels2), axis=0)
# X = np.load('diting_train_data.npy')
# y = np.load('diting_train_label.npy')
X = np.load('diting_train_data.npy')
y = np.load('diting_train_label.npy')



print(X.shape)


# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# 将数据集重塑为3D张量
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

y_train = np_utils.to_categorical(y_train, 3)
y_test = np_utils.to_categorical(y_test, 3)




# 创建卷积神经网络模型
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=2, activation=None, input_shape=(X_train.shape[1], 1)))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.25))
model.add(Conv1D(filters=32, kernel_size=2, activation=None))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.25))
model.add(Conv1D(filters=64, kernel_size=2, activation=None))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.25))
model.add(Conv1D(filters=128, kernel_size=2, activation=None))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.25))
model.add(Conv1D(filters=256, kernel_size=2, activation=None))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 编译模型
model.compile(optimizer=optimizers.Adam(learning_rate=0.001,decay=0.006), loss=keras.losses.categorical_crossentropy, metrics=['accuracy']) #0.001时候是最好的
model_checkpoint = callbacks.ModelCheckpoint('diting_model_test_8000_3.h5', monitor='val_accuracy', verbose=1, save_best_only=True)
# model_checkpoint = callbacks.ModelCheckpoint('diting_model_remain.h5', monitor='val_accuracy', verbose=1, save_best_only=True)
# 训练模型
history = model.fit(X_train, y_train, epochs=150, batch_size=8, validation_data=(X_test, y_test),
                   callbacks=[model_checkpoint])

# 保存模型
# save_model(model, 'diting_model_test_8000.h5')
with open('training_results_remain.txt', 'w') as file:
    file.write('Epoch\tTrain_Loss\tTrain_Acc\tTest_Loss\tTest_Acc\n')
    for i, (train_loss, train_acc, test_loss, test_acc) in enumerate(zip(history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy'])):
        file.write('{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(i+1, train_loss, train_acc, test_loss, test_acc))

# 载入模型
model = load_model('diting_model_test_8000_3.h5')
# model = load_model('diting_model_remain.h5')

# 输出验证集准确度和损失率
loss, accuracy = model.evaluate(X_test, y_test)
print('Validation Loss:', loss)
print('Validation Accuracy:', accuracy)

# 画出每个epoch的准确度和损失率
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()