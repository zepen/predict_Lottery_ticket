# -*- coding:utf-8 -*-
"""
Author: BigCat
"""
import time
import json
import argparse
import numpy as np
import pandas as pd
from config import *
from modeling import LstmWithCRFModel, SignalLstmModel, tf
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="ssq", type=str, help="选择训练数据: 双色球/大乐透")
parser.add_argument('--train_test_split', default=0.7, type=float, help="训练集占比, 设置大于0.5")
args = parser.parse_args()

pred_key = {}


def create_data(data, name, windows):
    """ 创建训练数据
    :param data: 数据集
    :param name: 玩法，双色球/大乐透
    :param windows: 训练窗口
    :return:
    """
    if not len(data):
        raise logger.error(" 请执行 get_data.py 进行数据下载！")
    else:
        # 创建模型文件夹
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        logger.info("训练数据已加载! ")

    data = data.iloc[:, 2:].values
    logger.info("训练集数据维度: {}".format(data.shape))
    x_data, y_data = [], []
    for i in range(len(data) - windows - 1):
        sub_data = data[i:(i+windows+1), :]
        x_data.append(sub_data[1:])
        y_data.append(sub_data[0])

    cut_num = 6 if name == "ssq" else 5
    return {
        "red": {
            "x_data": np.array(x_data)[:, :, :cut_num], "y_data": np.array(y_data)[:, :cut_num]
        },
        "blue": {
            "x_data": np.array(x_data)[:, :, cut_num:], "y_data": np.array(y_data)[:, cut_num:]
        }
    }


def create_train_test_data(name, windows, train_test_split):
    """ 划分数据集 """
    if train_test_split < 0.5:
        raise "训练集采样比例小于50%,训练终止,请求重新采样（train_test_split>0.5）!"
    path = "{}{}".format(name_path[name]["path"], data_file_name)
    data = pd.read_csv(path)
    logger.info("read data from path: {}".format(path))
    train_data = create_data(data.iloc[:int(len(data) * train_test_split)], name, windows)
    test_data = create_data(data.iloc[int(len(data) * train_test_split):], name, windows)
    logger.info(
        "train_data sample rate = {}, test_data sample rate = {}".format(train_test_split, round(1 - train_test_split, 2)))
    return train_data, test_data


def train_with_eval_red_ball_model(name, x_train, y_train, x_test, y_test):
    """ 红球模型训练与评估 """
    m_args = model_args[name]
    x_train = x_train - 1
    y_train = y_train - 1
    train_data_len = x_train.shape[0]
    logger.info("训练特征数据维度: {}".format(x_train.shape))
    logger.info("训练标签数据维度: {}".format(y_train.shape))

    x_test = x_test - 1
    y_test = y_test - 1
    test_data_len = x_test.shape[0]
    logger.info("测试特征数据维度: {}".format(x_test.shape))
    logger.info("测试标签数据维度: {}".format(y_test.shape))

    start_time = time.time()

    with tf.compat.v1.Session() as sess:
        red_ball_model = LstmWithCRFModel(
            batch_size=m_args["model_args"]["batch_size"],
            n_class=m_args["model_args"]["red_n_class"],
            ball_num=m_args["model_args"]["sequence_len"] if name == "ssq" else m_args["model_args"]["red_sequence_len"],
            w_size=m_args["model_args"]["windows_size"],
            embedding_size=m_args["model_args"]["red_embedding_size"],
            words_size=m_args["model_args"]["red_n_class"],
            hidden_size=m_args["model_args"]["red_hidden_size"],
            layer_size=m_args["model_args"]["red_layer_size"]
        )
        train_step = tf.compat.v1.train.AdamOptimizer(
            learning_rate=m_args["train_args"]["red_learning_rate"],
            beta1=m_args["train_args"]["red_beta1"],
            beta2=m_args["train_args"]["red_beta2"],
            epsilon=m_args["train_args"]["red_epsilon"],
            use_locking=False,
            name='Adam'
        ).minimize(red_ball_model.loss)
        sess.run(tf.compat.v1.global_variables_initializer())
        sequence_len = m_args["model_args"]["sequence_len"] \
            if name == "ssq" else m_args["model_args"]["red_sequence_len"]
        for epoch in range(m_args["model_args"]["red_epochs"]):
            for i in range(train_data_len):
                _, loss_, pred = sess.run([
                    train_step, red_ball_model.loss, red_ball_model.pred_sequence
                ], feed_dict={
                    "inputs:0": x_train[i:(i+1), :, :],
                    "tag_indices:0": y_train[i:(i+1), :],
                    "sequence_length:0": np.array([sequence_len]*1)
                })
                if i % 100 == 0:
                    logger.info("epoch: {}, loss: {}, tag: {}, pred: {}".format(
                        epoch, loss_, y_train[i:(i+1), :][0] + 1, pred[0] + 1)
                    )
        logger.info("训练耗时: {}".format(time.time() - start_time))
        pred_key[ball_name[0][0]] = red_ball_model.pred_sequence.name
        if not os.path.exists(m_args["path"]["red"]):
            os.makedirs(m_args["path"]["red"])
        saver = tf.compat.v1.train.Saver()
        saver.save(sess, "{}{}.{}".format(m_args["path"]["red"], red_ball_model_name, extension))
        logger.info("模型评估【{}】...".format(name_path[name]["name"]))
        eval_d = {}
        all_true_count = 0
        for j in range(test_data_len):
            true = y_test[j:(j + 1), :]
            pred = sess.run(red_ball_model.pred_sequence
                , feed_dict={
                    "inputs:0": x_test[j:(j + 1), :, :],
                    "sequence_length:0": np.array([sequence_len] * 1)
                })
            count = np.sum(true == pred + 1)
            all_true_count += count
            if count in eval_d:
                eval_d[count] += 1
            else:
                eval_d[count] = 1
        logger.info("测试期数: {}".format(test_data_len))
        for k, v in eval_d.items():
            logger.info("命中{}个球，{}期，占比: {}%".format(k, v, round(v * 100 / test_data_len, 2)))
        logger.info(
            "整体准确率: {}%".format(
                round(all_true_count * 100 / (test_data_len * sequence_len), 2)
            )
        )


def train_with_eval_blue_ball_model(name, x_train, y_train, x_test, y_test):
    """ 蓝球模型训练与评估 """
    m_args = model_args[name]
    x_train = x_train - 1
    train_data_len = x_train.shape[0]
    if name == "ssq":
        x_train = x_train.reshape(len(x_train), m_args["model_args"]["windows_size"])
        y_train = tf.keras.utils.to_categorical(y_train - 1, num_classes=m_args["model_args"]["blue_n_class"])
    else:
        y_train = y_train - 1
    logger.info("训练特征数据维度: {}".format(x_train.shape))
    logger.info("训练标签数据维度: {}".format(y_train.shape))

    x_test = x_test - 1
    test_data_len = x_test.shape[0]
    if name == "ssq":
        x_test = x_test.reshape(len(x_test), m_args["model_args"]["windows_size"])
        y_test = tf.keras.utils.to_categorical(y_test - 1, num_classes=m_args["model_args"]["blue_n_class"])
    else:
        y_test = y_test - 1
    logger.info("训练特征数据维度: {}".format(x_test.shape))
    logger.info("训练标签数据维度: {}".format(y_test.shape))

    start_time = time.time()

    with tf.compat.v1.Session() as sess:
        if name == "ssq":
            blue_ball_model = SignalLstmModel(
                batch_size=m_args["model_args"]["batch_size"],
                n_class=m_args["model_args"]["blue_n_class"],
                w_size=m_args["model_args"]["windows_size"],
                embedding_size=m_args["model_args"]["blue_embedding_size"],
                hidden_size=m_args["model_args"]["blue_hidden_size"],
                outputs_size=m_args["model_args"]["blue_n_class"],
                layer_size=m_args["model_args"]["blue_layer_size"]
            )
        else:
            blue_ball_model = LstmWithCRFModel(
                batch_size=m_args["model_args"]["batch_size"],
                n_class=m_args["model_args"]["blue_n_class"],
                ball_num=m_args["model_args"]["blue_sequence_len"],
                w_size=m_args["model_args"]["windows_size"],
                embedding_size=m_args["model_args"]["blue_embedding_size"],
                words_size=m_args["model_args"]["blue_n_class"],
                hidden_size=m_args["model_args"]["blue_hidden_size"],
                layer_size=m_args["model_args"]["blue_layer_size"]
            )
        train_step = tf.compat.v1.train.AdamOptimizer(
            learning_rate=m_args["train_args"]["blue_learning_rate"],
            beta1=m_args["train_args"]["blue_beta1"],
            beta2=m_args["train_args"]["blue_beta2"],
            epsilon=m_args["train_args"]["blue_epsilon"],
            use_locking=False,
            name='Adam'
        ).minimize(blue_ball_model.loss)
        sess.run(tf.compat.v1.global_variables_initializer())
        sequence_len = "" if name == "ssq" else m_args["model_args"]["blue_sequence_len"]
        for epoch in range(m_args["model_args"]["blue_epochs"]):
            for i in range(train_data_len):
                if name == "ssq":
                    _, loss_, pred = sess.run([
                        train_step, blue_ball_model.loss, blue_ball_model.pred_label
                    ], feed_dict={
                        "inputs:0": x_train[i:(i+1), :],
                        "tag_indices:0": y_train[i:(i+1), :],
                    })
                    if i % 100 == 0:
                        logger.info("epoch: {}, loss: {}, tag: {}, pred: {}".format(
                            epoch, loss_, np.argmax(y_train[i:(i+1), :][0]) + 1, pred[0] + 1)
                        )
                else:
                    _, loss_, pred = sess.run([
                        train_step, blue_ball_model.loss, blue_ball_model.pred_sequence
                    ], feed_dict={
                        "inputs:0": x_train[i:(i + 1), :, :],
                        "tag_indices:0": y_train[i:(i + 1), :],
                        "sequence_length:0": np.array([sequence_len] * 1)
                    })
                    if i % 100 == 0:
                        logger.info("epoch: {}, loss: {}, tag: {}, pred: {}".format(
                            epoch, loss_, y_train[i:(i + 1), :][0] + 1, pred[0] + 1)
                        )
        logger.info("训练耗时: {}".format(time.time() - start_time))
        pred_key[ball_name[1][0]] = blue_ball_model.pred_label.name if name == "ssq" else blue_ball_model.pred_sequence.name
        if not os.path.exists(m_args["path"]["blue"]):
            os.mkdir(m_args["path"]["blue"])
        saver = tf.compat.v1.train.Saver()
        saver.save(sess, "{}{}.{}".format(m_args["path"]["blue"], blue_ball_model_name, extension))
        logger.info("模型评估【{}】...".format(name_path[name]["name"]))
        eval_d = {}
        all_true_count = 0
        for j in range(test_data_len):
            if name == "ssq":
                true = y_test[j:(j + 1), :]
                pred = sess.run(blue_ball_model.pred_label
                , feed_dict={"inputs:0": x_test[j:(j + 1), :]})
            else:
                true = y_test[j:(j + 1), :]
                pred = sess.run(blue_ball_model.pred_sequence
                , feed_dict={
                    "inputs:0": x_test[j:(j + 1), :, :],
                    "sequence_length:0": np.array([sequence_len] * 1)
                })
            count = np.sum(true == pred + 1)
            all_true_count += count
            if count in eval_d:
                eval_d[count] += 1
            else:
                eval_d[count] = 1
        logger.info("测试期数: {}".format(test_data_len))
        for k, v in eval_d.items():
            logger.info("命中{}个球，{}期，占比: {}%".format(k, v, round(v * 100 / test_data_len, 2)))
        if name == "ssq":
            logger.info(
                "整体准确率: {}%".format(
                    round(all_true_count * 100 / test_data_len, 2)
                )
            )
        else:
            logger.info(
                "整体准确率: {}%".format(
                    round(all_true_count * 100 / (test_data_len * sequence_len), 2)
                )
            )


def run(name, train_test_split):
    """ 执行训练
    :param name: 玩法
    :param train_test_split: 训练集划分
    :return:
    """
    logger.info("正在创建【{}】训练集和测试集...".format(name_path[name]["name"]))
    train_data, test_data = create_train_test_data(
        name, model_args[name]["model_args"]["windows_size"], train_test_split
    )
    logger.info("开始训练【{}】红球模型...".format(name_path[name]["name"]))
    train_with_eval_red_ball_model(
        name,
        x_train=train_data["red"]["x_data"], y_train=train_data["red"]["y_data"],
        x_test=test_data["red"]["x_data"], y_test=test_data["red"]["y_data"],
    )

    tf.compat.v1.reset_default_graph()  # 重置网络图

    logger.info("开始训练【{}】蓝球模型...".format(name_path[name]["name"]))
    train_with_eval_blue_ball_model(
        name,
        x_train=train_data["blue"]["x_data"], y_train=train_data["blue"]["y_data"],
        x_test=test_data["blue"]["x_data"], y_test=test_data["blue"]["y_data"]
    )
    # 保存预测关键结点名
    with open("{}/{}/{}".format(model_path, name, pred_key_name), "w") as f:
        json.dump(pred_key, f)


if __name__ == '__main__':
    if not args.name:
        raise Exception("玩法名称不能为空！")
    else:
        run(args.name, args.train_test_split)
