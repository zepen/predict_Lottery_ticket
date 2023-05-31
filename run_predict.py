# -*- coding:utf-8 -*-
"""
Author: BigCat
"""
import argparse
import json
import time
import datetime
import numpy as np
import tensorflow as tf
from config import *
from get_data import get_current_number, spider
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="ssq", type=str, help="选择训练数据: 双色球/大乐透")
args = parser.parse_args()

# 关闭eager模式
tf.compat.v1.disable_eager_execution()


def load_model(name):
    """ 加载模型 """
    if name == "ssq":
        red_graph = tf.compat.v1.Graph()
        with red_graph.as_default():
            red_saver = tf.compat.v1.train.import_meta_graph(
                "{}red_ball_model.ckpt.meta".format(model_args[args.name]["path"]["red"])
            )
        red_sess = tf.compat.v1.Session(graph=red_graph)
        red_saver.restore(red_sess, "{}red_ball_model.ckpt".format(model_args[args.name]["path"]["red"]))
        logger.info("已加载红球模型！")

        blue_graph = tf.compat.v1.Graph()
        with blue_graph.as_default():
            blue_saver = tf.compat.v1.train.import_meta_graph(
                "{}blue_ball_model.ckpt.meta".format(model_args[args.name]["path"]["blue"])
            )
        blue_sess = tf.compat.v1.Session(graph=blue_graph)
        blue_saver.restore(blue_sess, "{}blue_ball_model.ckpt".format(model_args[args.name]["path"]["blue"]))
        logger.info("已加载蓝球模型！")

        # 加载关键节点名
        with open("{}/{}/{}".format(model_path, args.name, pred_key_name)) as f:
            pred_key_d = json.load(f)

        current_number = get_current_number(args.name)
        logger.info("【{}】最近一期:{}".format(name_path[args.name]["name"], current_number))
        return red_graph, red_sess, blue_graph, blue_sess, pred_key_d, current_number
    else:
        red_graph = tf.compat.v1.Graph()
        with red_graph.as_default():
            red_saver = tf.compat.v1.train.import_meta_graph(
                "{}red_ball_model.ckpt.meta".format(model_args[args.name]["path"]["red"])
            )
        red_sess = tf.compat.v1.Session(graph=red_graph)
        red_saver.restore(red_sess, "{}red_ball_model.ckpt".format(model_args[args.name]["path"]["red"]))
        logger.info("已加载红球模型！")

        blue_graph = tf.compat.v1.Graph()
        with blue_graph.as_default():
            blue_saver = tf.compat.v1.train.import_meta_graph(
                "{}blue_ball_model.ckpt.meta".format(model_args[args.name]["path"]["blue"])
            )
        blue_sess = tf.compat.v1.Session(graph=blue_graph)
        blue_saver.restore(blue_sess, "{}blue_ball_model.ckpt".format(model_args[args.name]["path"]["blue"]))
        logger.info("已加载蓝球模型！")

        # 加载关键节点名
        with open("{}/{}/{}".format(model_path,args.name , pred_key_name)) as f:
            pred_key_d = json.load(f)

        current_number = get_current_number(args.name)
        logger.info("【{}】最近一期:{}".format(name_path[args.name]["name"], current_number))
        return red_graph, red_sess, blue_graph, blue_sess, pred_key_d, current_number


def get_year():
    """ 截取年份
    eg：2020-->20, 2021-->21
    :return:
    """
    return int(str(datetime.datetime.now().year)[-2:])


def try_error(mode, name, predict_features, windows_size):
    """ 处理异常
    """
    if mode:
        return predict_features
    else:
        if len(predict_features) != windows_size:
            logger.warning("期号出现跳期，期号不连续！开始查找最近上一期期号！本期预测时间较久！")
            last_current_year = (get_year() - 1) * 1000
            max_times = 160
            while len(predict_features) != 3:
                predict_features = spider(name, last_current_year + max_times, get_current_number(name), "predict")[[x[0] for x in ball_name]]
                time.sleep(np.random.random(1).tolist()[0])
                max_times -= 1
            return predict_features
        return predict_features


def get_red_ball_predict_result(red_graph, red_sess, pred_key_d, predict_features, sequence_len, windows_size):
    """ 获取红球预测结果
    """
    name_list = [(ball_name[0], i + 1) for i in range(sequence_len)]
    data = predict_features[["{}_{}".format(name[0], i) for name, i in name_list]].values.astype(int) - 1
    with red_graph.as_default():
        reverse_sequence = tf.compat.v1.get_default_graph().get_tensor_by_name(pred_key_d[ball_name[0][0]])
        pred = red_sess.run(reverse_sequence, feed_dict={
            "inputs:0": data.reshape(1, windows_size, sequence_len),
            "sequence_length:0": np.array([sequence_len] * 1)
        })
    return pred, name_list


def get_blue_ball_predict_result(blue_graph, blue_sess, pred_key_d, name, predict_features, sequence_len, windows_size):
    """ 获取蓝球预测结果
    """
    if name == "ssq":
        data = predict_features[[ball_name[1][0]]].values.astype(int) - 1
        with blue_graph.as_default():
            softmax = tf.compat.v1.get_default_graph().get_tensor_by_name(pred_key_d[ball_name[1][0]])
            pred = blue_sess.run(softmax, feed_dict={
                "inputs:0": data.reshape(1, windows_size)
            })
        return pred
    else:
        name_list = [(ball_name[1], i + 1) for i in range(sequence_len)]
        data = predict_features[["{}_{}".format(name[0], i) for name, i in name_list]].values.astype(int) - 1
        with blue_graph.as_default():
            reverse_sequence = tf.compat.v1.get_default_graph().get_tensor_by_name(pred_key_d[ball_name[1][0]])
            pred = blue_sess.run(reverse_sequence, feed_dict={
                "inputs:0": data.reshape(1, windows_size, sequence_len),
                "sequence_length:0": np.array([sequence_len] * 1)
            })
        return pred, name_list


def get_final_result(red_graph, red_sess, blue_graph, blue_sess, pred_key_d, name, predict_features, mode=0):
    """" 最终预测函数
    """
    m_args = model_args[name]["model_args"]
    if name == "ssq":
        red_pred, red_name_list = get_red_ball_predict_result(
            red_graph, red_sess, pred_key_d,
            predict_features, m_args["sequence_len"], m_args["windows_size"]
        )
        blue_pred = get_blue_ball_predict_result(
            blue_graph, blue_sess, pred_key_d,
            name, predict_features, 0, m_args["windows_size"]
        )
        ball_name_list = ["{}_{}".format(name[mode], i) for name, i in red_name_list] + [ball_name[1][mode]]
        pred_result_list = red_pred[0].tolist() + blue_pred.tolist()
        return {
            b_name: int(res) + 1 for b_name, res in zip(ball_name_list, pred_result_list)
        }
    else:
        red_pred, red_name_list = get_red_ball_predict_result(
            red_graph, red_sess, pred_key_d,
            predict_features, m_args["red_sequence_len"], m_args["windows_size"]
        )
        blue_pred, blue_name_list = get_blue_ball_predict_result(
            blue_graph, blue_sess, pred_key_d,
            name, predict_features, m_args["blue_sequence_len"], m_args["windows_size"]
        )
        ball_name_list = ["{}_{}".format(name[mode], i) for name, i in red_name_list] + ["{}_{}".format(name[mode], i) for name, i in blue_name_list]
        pred_result_list = red_pred[0].tolist() + blue_pred[0].tolist()
        return {
            b_name: int(res) + 1 for b_name, res in zip(ball_name_list, pred_result_list)
        }


def run(name):
    """ 执行预测 """
    try:
        red_graph, red_sess, blue_graph, blue_sess, pred_key_d, current_number = load_model(name)
        windows_size = model_args[name]["model_args"]["windows_size"]
        data = spider(name, 1, current_number, "predict")
        logger.info("【{}】预测期号：{}".format(name_path[name]["name"], int(current_number) + 1))
        predict_features_ = try_error(1, name, data.iloc[:windows_size], windows_size)
        logger.info("预测结果：{}".format(get_final_result(
            red_graph, red_sess, blue_graph, blue_sess, pred_key_d, name, predict_features_))
        )
    except Exception as e:
        logger.info("模型加载失败，检查模型是否训练，错误：{}".format(e))


if __name__ == '__main__':
    if not args.name:
        raise Exception("玩法名称不能为空！")
    else:
        run(args.name)
