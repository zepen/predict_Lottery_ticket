# -*- coding:utf-8 -*-
"""
Author: Niuzepeng
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
from multiprocessing import Pool

DATA = pd.read_csv("data/data.csv")
COL_NAME = ["红球号码_1", "红球号码_2", "红球号码_3",
            "红球号码_4", "红球号码_5", "红球号码_6", "蓝球"]

X_DATA = []
Y_DATA = []


def transform_data(col_name):
    data_list = DATA[col_name].tolist()
    data_list_len = len(data_list)
    end_index = int(np.float(data_list_len / float(4)) * 4)
    data_list.reverse()
    fl = []
    for index, _ in enumerate(data_list[0: end_index - 4]):
        l = []
        for i in range(4):
            l.append(data_list[index + i])
        fl.append(l)
    return np.array(fl)


def create_model_data():
    for col_name in COL_NAME:
        data = transform_data(col_name)
        X_DATA.append(data[:, 0:3].reshape([-1, 3, 1]))
        Y_DATA.append(to_categorical(data[:, 3:].ravel()))


def train_model(x):
    n_class = Y_DATA[x].shape[1]
    model = Sequential()
    model.add(LSTM(10, input_shape=(3, 1), kernel_initializer='random_uniform'))
    model.add(Dense(n_class, activation="softmax"))
    sgd = SGD(lr=0.1)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])
    model.fit(X_DATA[x], Y_DATA[x], batch_size=1, epochs=5000, verbose=0)
    model.save("model/" + "model_" + str(x) + ".h5")

if __name__ == '__main__':
    create_model_data()
    pool = Pool(4)
    pool.map(train_model, range(7))
    pool.close()
    pool.join()
