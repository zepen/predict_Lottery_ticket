# -*- coding:utf-8 -*-
"""
Author: BigCat
"""
import time
import json
import numpy as np
import pandas as pd
from config import *
from modeling import *

pred_key = {}
DATA = pd.read_csv("{}{}".format(train_data_path, train_data_file))
if not len(DATA):
    raise Exception("[ERROR] 请执行 get_train_data.py 进行数据下载！")
else:
    # 创建模型文件夹
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    # 创建日志文件夹
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    # 创建日志文件
    if not os.path.exists(access_log):
        open(access_log,"w")
    if not os.path.exists(error_log):
        open(error_log,"w")
    print("[INFO] 训练数据已加载! ")


def create_train_data(name, windows, ball_num=6):
    """ 创建训练数据
    :param name: 红/蓝 球
    :param windows: 训练窗口
    :param ball_num: 多少颗球
    :return:
    """
    if name == BOLL_NAME[0][0]:
        data = DATA[["{}号码_{}".format(name, num + 1) for num in range(ball_num)]].values
    else:
        data = DATA[[name]].values
    print("[INFO] data shape: {}".format(data.shape))
    x_data, y_data = [], []
    for i in range(len(data) - windows - 1):
        sub_data = data[i:(i+windows+1), :]
        x_data.append(sub_data[1:])
        y_data.append(sub_data[0])
    if name == BOLL_NAME[0][0]:
        return np.array(x_data), np.array(y_data)
    else:
        return np.array(x_data).reshape(len(data) - windows - 1, windows), np.array(y_data)


def train_model(x_data, y_data, b_name):
    """ 模型训练
    :param x_data: 训练样本
    :param y_data: 训练标签
    :param b_name: 球颜色名
    :return:
    """
    if b_name == BOLL_NAME[0][0]:
        x_data = x_data - 1
        y_data = y_data - 1
        data_len = x_data.shape[0]
        print("[INFO] The x_data shape is {}".format(x_data.shape))
        print("[INFO] The y_data shape is {}".format(y_data.shape))
        with tf.compat.v1.Session() as sess:
            red_ball_model = RedBallModel(
                batch_size=batch_size,
                n_class=red_n_class,
                ball_num=sequence_len,
                w_size=windows_size,
                embedding_size=red_embedding_size,
                words_size=red_n_class,
                hidden_size=red_hidden_size,
                layer_size=red_layer_size
            )
            train_step = tf.compat.v1.train.AdamOptimizer(
                learning_rate=red_learning_rate,
                beta1=red_beta1,
                beta2=red_beta2,
                epsilon=red_epsilon,
                use_locking=False,
                name='Adam'
            ).minimize(red_ball_model.loss)
            sess.run(tf.compat.v1.global_variables_initializer())
            for epoch in range(red_epochs):
                for i in range(data_len):
                    _, loss_, pred = sess.run([
                        train_step, red_ball_model.loss, red_ball_model.pred_sequence
                    ], feed_dict={
                        "red_inputs:0": x_data[i:(i+1), :, :],
                        "red_tag_indices:0": y_data[i:(i+1), :],
                        "sequence_length:0": np.array([sequence_len]*1)
                    })
                    if i % 100 == 0:
                        print("[INFO] epoch: {}, loss: {}, tag: {}, pred: {}".format(
                            epoch, loss_, y_data[i:(i+1), :][0] + 1, pred[0] + 1)
                        )
            pred_key[b_name] = red_ball_model.pred_sequence.name
            if not os.path.exists(red_ball_model_path):
                os.mkdir(red_ball_model_path)
            saver = tf.compat.v1.train.Saver()
            saver.save(sess, "{}{}.{}".format(red_ball_model_path, red_ball_model_name, extension))
    elif b_name == BOLL_NAME[1][0]:
        # 重置网络图
        tf.compat.v1.reset_default_graph()
        x_data = x_data - 1
        data_len = x_data.shape[0]
        y_data = tf.keras.utils.to_categorical(y_data - 1, num_classes=blue_n_class)
        print("[INFO] The x_data shape is {}".format(x_data.shape))
        print("[INFO] The y_data shape is {}".format(y_data.shape))
        with tf.compat.v1.Session() as sess:
            blue_ball_model = BlueBallModel(
                batch_size=batch_size,
                n_class=blue_n_class,
                w_size=windows_size,
                embedding_size=blue_embedding_size,
                hidden_size=blue_hidden_size,
                outputs_size=blue_n_class,
                layer_size=blue_layer_size
            )
            train_step = tf.compat.v1.train.AdamOptimizer(
                learning_rate=blue_learning_rate,
                beta1=blue_beta1,
                beta2=blue_beta2,
                epsilon=blue_epsilon,
                use_locking=False,
                name='Adam'
            ).minimize(blue_ball_model.loss)
            sess.run(tf.compat.v1.global_variables_initializer())
            for epoch in range(blue_epochs):
                for i in range(data_len):
                    _, loss_, pred = sess.run([
                        train_step, blue_ball_model.loss, blue_ball_model.pred_label
                    ], feed_dict={
                        "blue_inputs:0": x_data[i:(i+1), :],
                        "blue_tag_indices:0": y_data[i:(i+1), :],
                    })
                    if i % 100 == 0:
                        print("[INFO] epoch: {}, loss: {}, tag: {}, pred: {}".format(
                            epoch, loss_, np.argmax(y_data[i:(i+1), :][0]) + 1, pred[0] + 1)
                        )
            pred_key[b_name] = blue_ball_model.pred_label.name
            if not os.path.exists(blue_ball_model_path):
                os.mkdir(blue_ball_model_path)
            saver = tf.compat.v1.train.Saver()
            saver.save(sess, "{}{}.{}".format(blue_ball_model_path, blue_ball_model_name, extension))
    # 保存预测关键结点名
    with open("{}/{}".format(model_path, pred_key_name), "w") as f:
        json.dump(pred_key, f)


if __name__ == '__main__':
    for b_n, _ in BOLL_NAME:
        start_time = time.time()
        print("[INFO] 开始训练: {}".format(b_n))
        x_train, y_train = create_train_data(b_n, windows_size)
        train_model(x_train, y_train, b_n)
        print("[INFO] 训练耗时: {}".format(time.time() - start_time))
