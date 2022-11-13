# -*- coding:utf-8 -*-
"""
Author: BigCat
"""
import json
import time
import datetime
import numpy as np
import tensorflow as tf
from config import *
from get_data import get_current_number, spider
from loguru import logger

# 关闭eager模式
tf.compat.v1.disable_eager_execution()

red_graph = tf.compat.v1.Graph()
with red_graph.as_default():
    red_saver = tf.compat.v1.train.import_meta_graph("{}red_ball_model.ckpt.meta".format(red_ball_model_path))
red_sess = tf.compat.v1.Session(graph=red_graph)
red_saver.restore(red_sess, "{}red_ball_model.ckpt".format(red_ball_model_path))
logger.info("已加载红球模型！")

blue_graph = tf.compat.v1.Graph()
with blue_graph.as_default():
    blue_saver = tf.compat.v1.train.import_meta_graph("{}blue_ball_model.ckpt.meta".format(blue_ball_model_path))
blue_sess = tf.compat.v1.Session(graph=blue_graph)
blue_saver.restore(blue_sess, "{}blue_ball_model.ckpt".format(blue_ball_model_path))
logger.info("已加载蓝球模型！")

# 加载关键节点名
with open("{}{}".format(model_path, pred_key_name)) as f:
    pred_key_d = json.load(f)

current_number = get_current_number()
logger.info("最近一期:{}".format(current_number))


def get_year():
    """ 截取年份
    eg：2020-->20, 2021-->21
    :return:
    """
    return int(str(datetime.datetime.now().year)[-2:])


def try_error(predict_features):
    """ 处理异常
    :param predict_features:
    :return:
    """
    if len(predict_features) != windows_size:
        logger.warning("期号出现跳期，期号不连续！开始查找最近上一期期号！本期预测时间较久！")
        last_current_year = (get_year() - 1) * 1000
        max_times = 160
        while len(predict_features) != 3:
            predict_features = spider(last_current_year + max_times, get_current_number(), "predict")[[x[0] for x in BOLL_NAME]]
            time.sleep(np.random.random(1).tolist()[0])
            max_times -= 1
        return predict_features
    return predict_features


def get_red_ball_predict_result(predict_features):
    """ 获取红球预测结果
    :param predict_features: 预测特征
    :return:
    """
    red_name_list = [(BOLL_NAME[0], i + 1) for i in range(sequence_len)]
    red_data = predict_features[["{}号码_{}".format(name[0], i) for name, i in red_name_list]].values.astype(int) - 1
    # 预测红球
    with red_graph.as_default():
        reverse_sequence = tf.compat.v1.get_default_graph().get_tensor_by_name(pred_key_d[BOLL_NAME[0][0]])
        red_pred = red_sess.run(reverse_sequence, feed_dict={
            "red_inputs:0": red_data.reshape(batch_size, windows_size, sequence_len),
            "sequence_length:0": np.array([sequence_len] * 1)
        })
    return red_pred, red_name_list


def get_blue_ball_predict_result(predict_features):
    """ 获取蓝球预测结果
    :return:
    """
    blue_data = predict_features[[BOLL_NAME[1][0]]].values.astype(int) - 1
    with blue_graph.as_default():
        softmax = tf.compat.v1.get_default_graph().get_tensor_by_name(pred_key_d[BOLL_NAME[1][0]])
        blue_pred = blue_sess.run(softmax, feed_dict={
            "blue_inputs:0": blue_data.reshape(batch_size, windows_size)
        })
    return blue_pred


def get_final_result(predict_features, mode=0):
    """" 最终预测函数
    :param predict_features: 预测特征
    :param mode: 模式，0：离线，1：在线api
    :return:
    """
    red_pred, red_name_list = get_red_ball_predict_result(predict_features)
    blue_pred = get_blue_ball_predict_result(predict_features)
    ball_name_list = ["{}_{}".format(name[mode], i) for name, i in red_name_list] + [BOLL_NAME[1][mode]]
    pred_result_list = red_pred[0].tolist() + blue_pred.tolist()
    return {
        b_name: int(res) + 1 for b_name, res in zip(ball_name_list, pred_result_list)
    }


if __name__ == '__main__':
    diff_number = windows_size - 1
    data = spider(str(int(current_number) - diff_number), current_number, "predict")
    logger.info("预测期号：{}".format(int(current_number) + 1))
    predict_features_ = try_error(data)
    logger.info("预测结果：{}".format(get_final_result(predict_features_)))
