# -*- coding:utf-8 -*-
"""
Author: BigCat
Modifier: KittenCN
"""
import time
import json
import argparse
import numpy as np
import pandas as pd
import warnings
from config import *
from modeling import LstmWithCRFModel, SignalLstmModel, tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from loguru import logger
import math

warnings.filterwarnings('ignore')
# tf.enable_eager_execution() # 开启动态图
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0],True)

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="ssq", type=str, help="选择训练数据")
parser.add_argument('--windows_size', default='3', type=str, help="训练窗口大小,如有多个，用'，'隔开")
parser.add_argument('--red_epochs', default=1, type=int, help="红球训练轮数")
parser.add_argument('--blue_epochs', default=1, type=int, help="蓝球训练轮数")
parser.add_argument('--batch_size', default=1, type=int, help="集合数量")
parser.add_argument('--predict_pro', default=0, type=int, help="更新batch_size")
args = parser.parse_args()

pred_key = {}
ori_data = None
save_epoch = 100

def create_train_data(name, windows):
    """ 创建训练数据
    :param name: 玩法，双色球/大乐透
    :param windows: 训练窗口
    :return:
    """
    global ori_data
    if ori_data is None:
        ori_data = pd.read_csv("{}{}".format(name_path[name]["path"], data_file_name))
    data = ori_data.copy()
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

    cut_num = 6
    if name == "ssq":
        cut_num = 6
    elif name == "dlt":
        cut_num = 5
    elif name == "pls":
        cut_num = 3
    elif name == "kl8":
        cut_num = 20
    return {
        "red": {
            "x_data": np.array(x_data)[:, :, :cut_num], "y_data": np.array(y_data)[:, :cut_num]
        },
        "blue": {
            "x_data": np.array(x_data)[:, :, cut_num:], "y_data": np.array(y_data)[:, cut_num:]
        }
    }


def train_red_ball_model(name, x_data, y_data):
    """ 红球模型训练
    :param name: 玩法
    :param x_data: 训练样本
    :param y_data: 训练标签
    :return:
    """
    m_args = model_args[name]
    if name not in ["pls"]:
        x_data = x_data - 1
        y_data = y_data - 1
    data_len = x_data.shape[0]
    logger.info("特征数据维度: {}".format(x_data.shape))
    logger.info("标签数据维度: {}".format(y_data.shape))
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
        syspath = model_path + model_args[args.name]["pathname"]['name'] + str(m_args["model_args"]["windows_size"]) + model_args[args.name]["subpath"]['red']
        if os.path.exists(syspath):
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, "{}red_ball_model.ckpt".format(syspath))
            logger.info("已加载红球模型！")

        dataset = tf.compat.v1.data.Dataset.from_tensor_slices((x_data, y_data))
        dataset = dataset.shuffle(buffer_size=data_len)
        dataset = dataset.batch(m_args["model_args"]["batch_size"])
        dataset = dataset.repeat(m_args["model_args"]["red_epochs"])
        iterator = dataset.make_one_shot_iterator()
        nextelement = iterator.get_next()
        index = 0
        epoch = 0
        epochindex = math.ceil(data_len / m_args["model_args"]["batch_size"])
        totalindex = epochindex * m_args["model_args"]["red_epochs"]
        epoch_start_time = time.time()
        perindex = 0
        totalloss = 0.0
        while True:
            try:
                x, y = sess.run(nextelement)
                batch_size = len(x)
                diff = m_args["model_args"]["batch_size"] - batch_size
                while diff > 0:
                    random_index = np.random.randint(0, data_len)
                    x = np.append(x, [x_data[random_index]], axis=0)
                    y = np.append(y, [y_data[random_index]], axis=0)
                    diff -= 1
                index += 1
                _, loss_, pred = sess.run([
                    train_step, red_ball_model.loss, red_ball_model.pred_sequence
                ], feed_dict={
                    "inputs:0": x,
                    "tag_indices:0": y,
                    "sequence_length:0": np.array([m_args["model_args"]["sequence_len"]]*m_args["model_args"]["batch_size"]) \
                        if name == "ssq" else np.array([m_args["model_args"]["red_sequence_len"]]*m_args["model_args"]["batch_size"])
                })
                perindex += 1
                totalloss += loss_
                if index % 10 == 0:
                    if name not in ["pls"]:
                        hotfixed = 1
                    else:
                        hotfixed = 0
                    logger.info("w_size: {}, index: {}, loss: {}, tag: {}, pred: {}".format(
                        str(m_args["model_args"]["windows_size"]), str(index) + '/' + str(totalindex), loss_, y[0] + hotfixed, pred[0] + hotfixed)
                    )
                    if args.predict_pro == 1:
                        pred_key[ball_name[0][0]] = red_ball_model.pred_sequence.name
                        if not os.path.exists(syspath):
                            os.makedirs(syspath)
                        saver = tf.compat.v1.train.Saver()
                        saver.save(sess, "{}{}.{}".format(syspath, red_ball_model_name, extension))
                        break
                if index % epochindex == 0:
                    epoch += 1
                    logger.info("epoch: {}, cost time: {}, ETA: {}, per_loss: {}".format(epoch, time.time() - epoch_start_time, (time.time() - epoch_start_time) * (m_args["model_args"]["red_epochs"] - epoch - 1), totalloss / perindex / m_args["model_args"]["batch_size"]))
                    epoch_start_time = time.time()
                    perindex = 0
                    totalloss = 0.0
                if epoch % save_epoch == 0 and epoch > 0:
                    pred_key[ball_name[0][0]] = red_ball_model.pred_sequence.name
                    if not os.path.exists(syspath):
                        os.makedirs(syspath)
                    saver = tf.compat.v1.train.Saver()
                    saver.save(sess, "{}{}.{}".format(syspath, red_ball_model_name, extension))
            except tf.errors.OutOfRangeError:
                logger.info("训练完成！")
                pred_key[ball_name[0][0]] = red_ball_model.pred_sequence.name
                if not os.path.exists(syspath):
                    os.makedirs(syspath)
                saver = tf.compat.v1.train.Saver()
                saver.save(sess, "{}{}.{}".format(syspath, red_ball_model_name, extension))
                break

def train_blue_ball_model(name, x_data, y_data):
    """ 蓝球模型训练
    :param name: 玩法
    :param x_data: 训练样本
    :param y_data: 训练标签
    :return:
    """
    m_args = model_args[name]
    x_data = x_data - 1
    y_data = y_data - 1
    data_len = x_data.shape[0]
    if name == "ssq":
        x_data = x_data.reshape(len(x_data), m_args["model_args"]["windows_size"])
        y_data = tf.keras.utils.to_categorical(y_data, num_classes=m_args["model_args"]["blue_n_class"])
    logger.info("特征数据维度: {}".format(x_data.shape))
    logger.info("标签数据维度: {}".format(y_data.shape))
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
        syspath = model_path + model_args[args.name]["pathname"]['name'] + str(m_args["model_args"]["windows_size"]) + model_args[args.name]["subpath"]['blue']
        if os.path.exists(syspath):
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, "{}blue_ball_model.ckpt".format(syspath))
            logger.info("已加载蓝球模型！")
        
        dataset = tf.compat.v1.data.Dataset.from_tensor_slices((x_data, y_data))
        dataset = dataset.shuffle(buffer_size=data_len)
        dataset = dataset.batch(m_args["model_args"]["batch_size"])
        dataset = dataset.repeat(m_args["model_args"]["blue_epochs"])
        iterator = dataset.make_one_shot_iterator()
        nextelement = iterator.get_next()
        index = 0
        epoch = 0
        epochindex = math.ceil(data_len / m_args["model_args"]["batch_size"])
        totalindex = epochindex * m_args["model_args"]["blue_epochs"]
        epoch_start_time = time.time()
        perindex = 0
        totalloss = 0.0
        while True:
            try:
                x, y = sess.run(nextelement)
                batch_size = len(x)
                diff = m_args["model_args"]["batch_size"] - batch_size
                while diff > 0:
                    random_index = np.random.randint(0, data_len)
                    x = np.append(x, [x_data[random_index]], axis=0)
                    y = np.append(y, [y_data[random_index]], axis=0)
                    diff -= 1
                index += 1
                if name == "ssq":
                    _, loss_, pred = sess.run([
                        train_step, blue_ball_model.loss, blue_ball_model.pred_label
                    ], feed_dict={
                        "inputs:0": x,
                        "tag_indices:0": y,
                    })
                    perindex += 1
                    totalloss += loss_
                    if index % 10 == 0:
                        logger.info("w_size: {}, epoch: {}, loss: {}, tag: {}, pred: {}".format(
                            str(m_args["model_args"]["windows_size"]), str(index) + '/' + str(totalindex), loss_, np.argmax(y[0]) + 1, pred[0] + 1)
                        )
                        if args.predict_pro == 1:
                            pred_key[ball_name[1][0]] = blue_ball_model.pred_label.name if name == "ssq" else blue_ball_model.pred_sequence.name
                            if not os.path.exists(syspath):
                                os.mkdir(syspath)
                            saver = tf.compat.v1.train.Saver()
                            saver.save(sess, "{}{}.{}".format(syspath, blue_ball_model_name, extension))
                            break
                else:
                    _, loss_, pred = sess.run([
                        train_step, blue_ball_model.loss, blue_ball_model.pred_sequence
                    ], feed_dict={
                        "inputs:0": x,
                        "tag_indices:0": y,
                        "sequence_length:0": np.array([m_args["model_args"]["blue_sequence_len"]] * m_args["model_args"]["batch_size"])
                    })
                    perindex += 1
                    totalloss += loss_
                    if index % 10 == 0:
                        logger.info("w_size: {}, epoch: {}, loss: {}, tag: {}, pred: {}".format(
                            str(m_args["model_args"]["windows_size"]), str(index) + '/' + str(totalindex), loss_,y[0] + 1, pred[0] + 1)
                        )
                        if args.predict_pro == 1:
                            pred_key[ball_name[1][0]] = blue_ball_model.pred_label.name if name == "ssq" else blue_ball_model.pred_sequence.name
                            if not os.path.exists(syspath):
                                os.mkdir(syspath)
                            saver = tf.compat.v1.train.Saver()
                            saver.save(sess, "{}{}.{}".format(syspath, blue_ball_model_name, extension))
                            break
                if index % epochindex == 0:
                    epoch += 1
                    logger.info("epoch: {}, cost time: {}, ETA: {}, per_loss: {}".format(epoch, time.time() - epoch_start_time, (time.time() - epoch_start_time) * (m_args["model_args"]["blue_epochs"] - epoch - 1), totalloss / perindex / m_args["model_args"]["batch_size"]))
                    epoch_start_time = time.time()
                    perindex = 0
                    totalloss = 0.0
                if epoch % save_epoch == 0 and epoch > 0:
                    pred_key[ball_name[1][0]] = blue_ball_model.pred_label.name if name == "ssq" else blue_ball_model.pred_sequence.name
                    if not os.path.exists(syspath):
                        os.mkdir(syspath)
                    saver = tf.compat.v1.train.Saver()
                    saver.save(sess, "{}{}.{}".format(syspath, blue_ball_model_name, extension))
                    
            except tf.errors.OutOfRangeError:
                logger.info("训练完成！")
                pred_key[ball_name[1][0]] = blue_ball_model.pred_label.name if name == "ssq" else blue_ball_model.pred_sequence.name
                if not os.path.exists(syspath):
                    os.mkdir(syspath)
                saver = tf.compat.v1.train.Saver()
                saver.save(sess, "{}{}.{}".format(syspath, blue_ball_model_name, extension))
                break

def action(name):
    tf.compat.v1.reset_default_graph()

    logger.info("正在创建【{}】数据集...".format(name_path[name]["name"]))
    train_data = create_train_data(args.name, model_args[name]["model_args"]["windows_size"])

    if model_args[name]["model_args"]["red_epochs"] > 0:
        logger.info("开始训练【{}】红球模型...".format(name_path[name]["name"]))
        start_time = time.time()
        train_red_ball_model(name, x_data=train_data["red"]["x_data"], y_data=train_data["red"]["y_data"])
        logger.info("训练耗时: {}".format(time.time() - start_time))

    if name not in ["pls", "kl8"] and model_args[name]["model_args"]["blue_epochs"] > 0:
        tf.compat.v1.reset_default_graph()  # 重置网络图

        logger.info("开始训练【{}】蓝球模型...".format(name_path[name]["name"]))
        start_time = time.time()
        train_blue_ball_model(name, x_data=train_data["blue"]["x_data"], y_data=train_data["blue"]["y_data"])
        logger.info("训练耗时: {}".format(time.time() - start_time))

    # 保存预测关键结点名
    with open("{}/{}/{}".format(model_path, name, pred_key_name), "w") as f:
        json.dump(pred_key, f)

def run(name, windows_size):
    """ 执行训练
    :param name: 玩法
    :return:
    """
    if int(windows_size[0]) == 0:
        action(name)
    else:
        for size in windows_size:
            model_args[name]["model_args"]["windows_size"] = int(size)
            action(name)

if __name__ == '__main__':
    list_windows_size = args.windows_size.split(",")
    if not args.name:
        raise Exception("玩法名称不能为空！")
    elif not args.windows_size:
        raise Exception("窗口大小不能为空！")
    else:
        model_args[args.name]["model_args"]["red_epochs"] = int(args.red_epochs)
        model_args[args.name]["model_args"]["blue_epochs"] = int(args.blue_epochs)
        model_args[args.name]["model_args"]["batch_size"] = int(args.batch_size)
        if args.predict_pro == 1:
            list_windows_size = [3,5,10,30,50,100,300,500]
            model_args[args.name]["model_args"]["red_epochs"] = 1
            model_args[args.name]["model_args"]["blue_epochs"] = 1
            model_args[args.name]["model_args"]["batch_size"] = 1
        run(args.name, list_windows_size)
