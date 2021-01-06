# -*- coding:utf-8 -*-
"""
Author: BigCat
"""
import os
import time
import numpy as np
import pandas as pd
from config import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

DATA = pd.read_csv("{}{}".format(train_data_path, train_data_file))
if not len(DATA):
    raise Exception("[ERROR] 请执行 get_train_data.py 进行数据下载！")
else:
    print("[INFO] 训练数据已加载! ")


def transform_data(name):
    """ 数据转换
    :param name: 要训练的球号
    :return:
    """
    data_list = DATA[name].tolist()
    data_list_len = len(data_list)
    end_index = int(np.float(data_list_len / float(4)) * 4)
    data_list.reverse()
    fl = []
    for index, _ in enumerate(data_list[0: end_index - 4]):
        l_ = []
        for i in range(4):
            l_.append(data_list[index + i])
        fl.append(l_)
    return np.array(fl)


def create_model_data(name, windows):
    """ 创建训练数据
    :param name: 要训练的球号
    :param windows: 训练窗口
    :return:
    """
    data = transform_data(name)
    x_data = data[:, 0:windows].reshape([-1, windows, 1])
    y_data = data[:, windows:].ravel()
    return x_data, y_data


def build_model(**kwargs):
    """ 构建模型
    :return:
    """
    hidden_size, outputs_size = kwargs["hidden_size"], kwargs["outputs_size"]
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(3, 1), return_sequences=True))
    model.add(LSTM(hidden_size, recurrent_dropout=0.2))
    model.add(Dense(outputs_size, activation="softmax"))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=["accuracy"])
    return model


def train_model(x_data, y_data, b_name):
    """ 模型训练
    :param x_data: 训练样本
    :param y_data: 训练标签
    :param b_name: 球号名
    :return:
    """
    n_class = 0
    if b_name[0] == "红":
        n_class = 33
    elif b_name[0] == "蓝":
        n_class = 16
    x_data = x_data - 1
    y_data = to_categorical(y_data - 1, num_classes=n_class)
    print("[INFO] The x_data shape is {}".format(x_data.shape))
    print("[INFO] The y_data shape is {}".format(y_data.shape))
    model = build_model(hidden_size=32, outputs_size=n_class)
    callbacks = [
        EarlyStopping(monitor='accuracy', patience=3, verbose=2, mode='max')
    ]
    model.fit(x_data, y_data, batch_size=1, epochs=100, verbose=1, callbacks=callbacks)
    if not os.path.exists("model"):
        os.mkdir("model")
    model.save("model/lstm_model_{}.h5".format(b_name))


if __name__ == '__main__':
    windows_size = 3
    for b_n in BOLL_NAME:
        start_time = time.time()
        print("[INFO] 开始训练: {}".format(b_n))
        x_train, y_train = create_model_data(b_n, windows_size)
        train_model(x_train, y_train, b_n)
        print("[INFO] 训练耗时: {}".format(time.time() - start_time))
