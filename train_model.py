# -*- coding:utf-8 -*-
"""
Author: BigCat
"""
import os
import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import config
from modeling import BlueBallModel, RedBallModel

pred_key = {}
with open('{}{}'.format(config.train_data_path, config.train_data_file), 'r', encoding='utf-8') as f:
    data_raw = json.load(f)
DATA = pd.DataFrame(data_raw)
if not len(DATA):
    raise Exception("[ERROR] 请执行 get_train_data.py 进行数据下载！")
else:
    print("[INFO] 训练数据已加载! ")


def create_train_data(name, windows, ball_num=6):
    """创建训练数据
    :param name: 红/蓝 球
    :param windows: 训练窗口
    :param ball_num: 多少颗球
    :return:
    """
    if name == config.BOLL_NAME[0]:
        data = DATA[["{}号码_{}".format(name, num + 1) for num in range(ball_num)]].values.astype(int)
    else:
        data = DATA[[name]].values.astype(int)
    print("[INFO] data shape: {}".format(data.shape))
    x_data, y_data = [], []
    for i in range(len(data) - windows - 1):
        sub_data = data[i: (i + windows + 1), :]
        x_data.append(sub_data[1:])
        y_data.append(sub_data[0])
    if name == config.BOLL_NAME[0]:
        return np.array(x_data), np.array(y_data)
    else:
        return np.array(x_data).reshape(len(data) - windows - 1, windows), np.array(y_data)


def train_model(x_data, y_data, b_name):
    """模型训练
    :param x_data: 训练样本
    :param y_data: 训练标签
    :param b_name: 球颜色名
    :return:
    """
    if b_name == config.BOLL_NAME[0]:
        x_data = x_data - 1
        y_data = y_data - 1
        data_len = x_data.shape[0]
        print("[INFO] The x_data shape is {}".format(x_data.shape))
        print("[INFO] The y_data shape is {}".format(y_data.shape))
        with tf.compat.v1.Session() as sess:
            red_ball_model = RedBallModel(
                batch_size=config.batch_size,
                n_class=config.red_n_class,
                ball_num=config.sequence_len,
                w_size=config.windows_size,
                embedding_size=config.red_embedding_size,
                words_size=config.red_n_class,
                hidden_size=config.red_hidden_size,
                layer_size=config.red_layer_size,
            )
            train_step = tf.compat.v1.train.AdamOptimizer(
                learning_rate=config.red_learning_rate,
                beta1=config.red_beta1,
                beta2=config.red_beta2,
                epsilon=config.red_epsilon,
                use_locking=False,
                name='Adam',
            ).minimize(red_ball_model.loss)
            sess.run(tf.compat.v1.global_variables_initializer())
            for epoch in range(config.red_epochs):
                for i in range(data_len):
                    _, loss_, pred = sess.run(
                        [train_step, red_ball_model.loss, red_ball_model.pred_sequence],
                        feed_dict={
                            "red_inputs:0": x_data[i: (i + 1), :, :],
                            "red_tag_indices:0": y_data[i: (i + 1), :],
                            "sequence_length:0": np.array([config.sequence_len] * 1),
                        },
                    )
                    if i % 100 == 0:
                        print(
                            "[INFO] epoch: {}, loss: {}, tag: {}, pred: {}".format(
                                epoch, loss_, y_data[i: (i + 1), :][0] + 1, pred[0] + 1
                            )
                        )
            pred_key[b_name] = red_ball_model.pred_sequence.name
            if not os.path.exists(config.red_ball_model_path):
                os.mkdir(config.red_ball_model_path)
            saver = tf.compat.v1.train.Saver()
            saver.save(sess, "{}{}.{}".format(config.red_ball_model_path, config.red_ball_model_name, config.extension))
    elif b_name == config.BOLL_NAME[1]:
        # 重置网络图
        tf.compat.v1.reset_default_graph()
        x_data = x_data - 1
        data_len = x_data.shape[0]
        y_data = tf.keras.utils.to_categorical(y_data - 1, num_classes=config.blue_n_class)
        print("[INFO] The x_data shape is {}".format(x_data.shape))
        print("[INFO] The y_data shape is {}".format(y_data.shape))
        with tf.compat.v1.Session() as sess:
            blue_ball_model = BlueBallModel(
                batch_size=config.batch_size,
                n_class=config.blue_n_class,
                w_size=config.windows_size,
                embedding_size=config.blue_embedding_size,
                hidden_size=config.blue_hidden_size,
                outputs_size=config.blue_n_class,
                layer_size=config.blue_layer_size,
            )
            train_step = tf.compat.v1.train.AdamOptimizer(
                learning_rate=config.blue_learning_rate,
                beta1=config.blue_beta1,
                beta2=config.blue_beta2,
                epsilon=config.blue_epsilon,
                use_locking=False,
                name='Adam',
            ).minimize(blue_ball_model.loss)
            sess.run(tf.compat.v1.global_variables_initializer())
            for epoch in range(config.blue_epochs):
                for i in range(data_len):
                    _, loss_, pred = sess.run(
                        [train_step, blue_ball_model.loss, blue_ball_model.pred_label],
                        feed_dict={
                            "blue_inputs:0": x_data[i: (i + 1), :],
                            "blue_tag_indices:0": y_data[i: (i + 1), :],
                        },
                    )
                    if i % 100 == 0:
                        print(
                            "[INFO] epoch: {}, loss: {}, tag: {}, pred: {}".format(
                                epoch,
                                loss_,
                                np.argmax(y_data[i: (i + 1), :][0]) + 1,
                                pred[0] + 1,
                            )
                        )
            pred_key[b_name] = blue_ball_model.pred_label.name
            if not os.path.exists(config.blue_ball_model_path):
                os.mkdir(config.blue_ball_model_path)
            saver = tf.compat.v1.train.Saver()
            saver.save(sess, "{}{}.{}".format(config.blue_ball_model_path, config.blue_ball_model_name, config.extension))
    # 保存预测关键结点名
    with open("{}{}".format(config.model_path, config.pred_key_name), "w") as f:
        json.dump(pred_key, f)


if __name__ == '__main__':
    for b_n in config.BOLL_NAME:
        start_time = time.time()
        print("[INFO] 开始训练: {}".format(b_n))
        x_train, y_train = create_train_data(b_n, config.windows_size)
        train_model(x_train, y_train, b_n)
        print("[INFO] 训练耗时: {}".format(time.time() - start_time))
