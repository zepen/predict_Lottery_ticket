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
# from get_data import get_current_number, spider
from loguru import logger
import requests
import pandas as pd
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="ssq", type=str, help="选择训练数据: 双色球/大乐透")
parser.add_argument('--windows_size', default='3', type=str, help="训练窗口大小,如有多个，用'，'隔开")
args = parser.parse_args()

def get_url(name):
    """
    :param name: 玩法名称
    :return:
    """
    url = "https://datachart.500.com/{}/history/".format(name)
    path = "newinc/history.php?start={}&end="
    return url, path

def get_current_number(name):
    """ 获取最新一期数字
    :return: int
    """
    url, _ = get_url(name)
    r = requests.get("{}{}".format(url, "history.shtml"), verify=False)
    r.encoding = "gb2312"
    soup = BeautifulSoup(r.text, "lxml")
    current_num = soup.find("div", class_="wrap_datachart").find("input", id="end")["value"]
    return current_num


def spider(name, start, end, mode):
    """ 爬取历史数据
    :param name 玩法
    :param start 开始一期
    :param end 最近一期
    :param mode 模式，train：训练模式，predict：预测模式（训练模式会保持文件）
    :return:
    """
    url, path = get_url(name)
    url = "{}{}{}".format(url, path.format(start), end)
    r = requests.get(url=url, verify=False)
    r.encoding = "gb2312"
    soup = BeautifulSoup(r.text, "lxml")
    trs = soup.find("tbody", attrs={"id": "tdata"}).find_all("tr")
    data = []
    for tr in trs:
        item = dict()
        if name == "ssq":
            item[u"期数"] = tr.find_all("td")[0].get_text().strip()
            for i in range(6):
                item[u"红球_{}".format(i+1)] = tr.find_all("td")[i+1].get_text().strip()
            item[u"蓝球"] = tr.find_all("td")[7].get_text().strip()
            data.append(item)
        elif name == "dlt":
            item[u"期数"] = tr.find_all("td")[0].get_text().strip()
            for i in range(5):
                item[u"红球_{}".format(i+1)] = tr.find_all("td")[i+1].get_text().strip()
            for j in range(2):
                item[u"蓝球_{}".format(j+1)] = tr.find_all("td")[6+j].get_text().strip()
            data.append(item)
        else:
            logger.warning("抱歉，没有找到数据源！")

    if mode == "train":
        df = pd.DataFrame(data)
        df.to_csv("{}{}".format(name_path[name]["path"], data_file_name), encoding="utf-8")
        return pd.DataFrame(data)
    elif mode == "predict":
        return pd.DataFrame(data)

# 关闭eager模式
tf.compat.v1.disable_eager_execution()

red_graph = tf.compat.v1.Graph()
blue_graph = tf.compat.v1.Graph()
pred_key_d = {}
red_sess = tf.compat.v1.Session(graph=red_graph)
blue_sess = tf.compat.v1.Session(graph=blue_graph)
current_number = get_current_number(args.name)

def run_predict(window_size):
    global pred_key_d, red_graph, blue_graph, red_sess, blue_sess, current_number
    if window_size != 0:
        model_args[args.name]["model_args"]["windows_size"] = window_size
    redpath = model_path + model_args[args.name]["pathname"]['name'] + str(model_args[args.name]["model_args"]["windows_size"]) + model_args[args.name]["subpath"]['red']
    bluepath = model_path + model_args[args.name]["pathname"]['name'] + str(model_args[args.name]["model_args"]["windows_size"]) + model_args[args.name]["subpath"]['blue']
    if args.name == "ssq":
        red_graph = tf.compat.v1.Graph()
        with red_graph.as_default():
            red_saver = tf.compat.v1.train.import_meta_graph(
                "{}red_ball_model.ckpt.meta".format(redpath)
            )
        red_sess = tf.compat.v1.Session(graph=red_graph)
        red_saver.restore(red_sess, "{}red_ball_model.ckpt".format(redpath))
        logger.info("已加载红球模型！窗口大小:{}".format(model_args[args.name]["model_args"]["windows_size"]))

        blue_graph = tf.compat.v1.Graph()
        with blue_graph.as_default():
            blue_saver = tf.compat.v1.train.import_meta_graph(
                "{}blue_ball_model.ckpt.meta".format(bluepath)
            )
        blue_sess = tf.compat.v1.Session(graph=blue_graph)
        blue_saver.restore(blue_sess, "{}blue_ball_model.ckpt".format(bluepath))
        logger.info("已加载蓝球模型！窗口大小:{}".format(model_args[args.name]["model_args"]["windows_size"]))

        # 加载关键节点名
        with open("{}/{}/{}".format(model_path, args.name, pred_key_name)) as f:
            pred_key_d = json.load(f)

        current_number = get_current_number(args.name)
        logger.info("【{}】最近一期:{}".format(name_path[args.name]["name"], current_number))

    else:
        red_graph = tf.compat.v1.Graph()
        with red_graph.as_default():
            red_saver = tf.compat.v1.train.import_meta_graph(
                "{}red_ball_model.ckpt.meta".format(redpath)
            )
        red_sess = tf.compat.v1.Session(graph=red_graph)
        red_saver.restore(red_sess, "{}red_ball_model.ckpt".format(redpath))
        logger.info("已加载红球模型！窗口大小:{}".format(model_args[args.name]["model_args"]["windows_size"]))

        blue_graph = tf.compat.v1.Graph()
        with blue_graph.as_default():
            blue_saver = tf.compat.v1.train.import_meta_graph(
                "{}blue_ball_model.ckpt.meta".format(bluepath)
            )
        blue_sess = tf.compat.v1.Session(graph=blue_graph)
        blue_saver.restore(blue_sess, "{}blue_ball_model.ckpt".format(bluepath))
        logger.info("已加载蓝球模型！窗口大小:{}".format(model_args[args.name]["model_args"]["windows_size"]))

        # 加载关键节点名
        with open("{}/{}/{}".format(model_path,args.name , pred_key_name)) as f:
            pred_key_d = json.load(f)

        current_number = get_current_number(args.name)
        logger.info("【{}】最近一期:{}".format(name_path[args.name]["name"], current_number))


def get_year():
    """ 截取年份
    eg：2020-->20, 2021-->21
    :return:
    """
    return int(str(datetime.datetime.now().year)[-2:])


def try_error(name, predict_features, windows_size):
    """ 处理异常
    """
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


def get_red_ball_predict_result(predict_features, sequence_len, windows_size):
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


def get_blue_ball_predict_result(name, predict_features, sequence_len, windows_size):
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


def get_final_result(name, predict_features, mode=0):
    """" 最终预测函数
    """
    m_args = model_args[name]["model_args"]
    if name == "ssq":
        red_pred, red_name_list = get_red_ball_predict_result(predict_features, m_args["sequence_len"], m_args["windows_size"])
        blue_pred = get_blue_ball_predict_result(name, predict_features, 0, m_args["windows_size"])
        ball_name_list = ["{}_{}".format(name[mode], i) for name, i in red_name_list] + [ball_name[1][mode]]
        pred_result_list = red_pred[0].tolist() + blue_pred.tolist()
        return {
            b_name: int(res) + 1 for b_name, res in zip(ball_name_list, pred_result_list)
        }
    else:
        red_pred, red_name_list = get_red_ball_predict_result(predict_features, m_args["red_sequence_len"], m_args["windows_size"])
        blue_pred, blue_name_list = get_blue_ball_predict_result(name, predict_features, m_args["blue_sequence_len"], m_args["windows_size"])
        ball_name_list = ["{}_{}".format(name[mode], i) for name, i in red_name_list] + ["{}_{}".format(name[mode], i) for name, i in blue_name_list]
        pred_result_list = red_pred[0].tolist() + blue_pred[0].tolist()
        return {
            b_name: int(res) + 1 for b_name, res in zip(ball_name_list, pred_result_list)
        }


def run(name):
    windows_size = model_args[name]["model_args"]["windows_size"]
    diff_number = windows_size - 1
    data = spider(name, str(int(current_number) - diff_number), current_number, "predict")
    logger.info("【{}】预测期号：{} 窗口大小:{}".format(name_path[name]["name"], int(current_number) + 1, windows_size))
    predict_features_ = try_error(name, data, windows_size)
    logger.info("预测结果：{}".format(get_final_result(name, predict_features_)))


if __name__ == '__main__':
    if not args.name:
        raise Exception("玩法名称不能为空！")
    elif not args.windows_size:
        raise Exception("窗口大小不能为空！")
    else:
        list_windows_size = args.windows_size.split(",")
        for size in list_windows_size:
            tf.compat.v1.reset_default_graph()
            run_predict(int(size))
            run(args.name)
        
