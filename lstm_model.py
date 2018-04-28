# -*- coding:utf-8 -*-
"""
Author: Niuzepeng
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

DATA = pd.read_csv("data/data.csv")
COL_NAME = ["红球号码_1", "红球号码_2", "红球号码_3",
            "红球号码_4", "红球号码_5", "红球号码_6", "蓝球"]


def transform_data():
    data_list = []
    for col_name in COL_NAME:
        data_list.extend(DATA[col_name].tolist())
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
    data = transform_data()
    x_data = data[:, 0:3].reshape([-1, 3, 1])
    y_data = to_categorical(data[:, 3:].ravel())
    return x_data, y_data


def train_model(x_data, y_data):
    n_class = y_data.shape[1]
    print("The x_data shape is {}".format(x_data.shape))
    print("The y_data shape is {}".format(y_data.shape))
    model = Sequential()
    model.add(LSTM(32, input_shape=(3, 1), return_sequences=True))
    model.add(LSTM(32, recurrent_dropout=0.2))
    model.add(Dense(n_class, activation="softmax"))
    adam = Adam()
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=["accuracy"])
    model.fit(x_data, y_data, batch_size=1, epochs=10000, verbose=1)
    model.save("model/" + "lstm_model.h5")

if __name__ == '__main__':
    x_train, y_train = create_model_data()
    train_model(x_train, y_train)
