import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
from multiprocessing import Pool

# 1号球
# 2号球
# 3号球
# 4号球
# 5号球
# 6号球
# 蓝球
x_data = np.array([[[1], [2], [3]], [[2], [3], [4]], [[3], [4], [5]]])
x_data2 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
y_data = np.array([0, 1, 2])
y_data = to_categorical(y_data)


def train_model(x):
    model = Sequential()
    model.add(LSTM(10, input_shape=(3, 1), kernel_initializer='random_uniform'))
    model.add(Dense(3, activation="softmax"))
    sgd = SGD(lr=0.1)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])
    model.fit(x_data[x], y_data[x], batch_size=1, epochs=5000, verbose=0)
    model.save("model/" + "model_" + str(x))

if __name__ == '__main__':
    pool = Pool(4)
    pool.map(train_model, range(7))
    pool.close()
    pool.join()
