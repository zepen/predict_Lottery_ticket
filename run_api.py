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
from flask import Flask
from get_train_data import get_current_number, spider, pd

# 关闭eager模式
tf.compat.v1.disable_eager_execution()

red_graph = tf.compat.v1.Graph()
with red_graph.as_default():
    red_saver = tf.compat.v1.train.import_meta_graph("{}red_ball_model.ckpt.meta".format(red_ball_model_path))
red_sess = tf.compat.v1.Session(graph=red_graph)
red_saver.restore(red_sess, "{}red_ball_model.ckpt".format(red_ball_model_path))
print("[INFO] 已加载红球模型！")

blue_graph = tf.compat.v1.Graph()
with blue_graph.as_default():
    blue_saver = tf.compat.v1.train.import_meta_graph("{}blue_ball_model.ckpt.meta".format(blue_ball_model_path))
blue_sess = tf.compat.v1.Session(graph=blue_graph)
blue_saver.restore(blue_sess, "{}blue_ball_model.ckpt".format(blue_ball_model_path))
print("[INFO] 已加载蓝球模型！")

# 加载关键节点名
with open("{}{}".format(model_path, pred_key_name)) as f:
    pred_key_d = json.load(f)

app = Flask(__name__)


def get_year():
    """ 截取年份
    eg：2020-->20, 2021-->21
    :return:
    """
    return int(str(datetime.datetime.now().year)[-2:])


@app.route('/')
def main():
    return "Welcome to use!"


@app.route('/predict_api', methods=['GET'])
def get_predict_result():
    diff_number = windows_size - 1
    data = spider(str(int(get_current_number()) - diff_number), get_current_number(), "predict")
    red_name_list = [(BOLL_NAME[0], i + 1) for i in range(sequence_len)]
    red_data = data[["{}号码_{}".format(name[0], i) for name, i in red_name_list]].values.astype(int) - 1
    blue_data = data[[BOLL_NAME[1][0]]].values.astype(int) - 1
    if len(data) != 3:
        print("[WARN] 期号出现跳期，期号不连续！开始查找最近上一期期号！本期预测时间较久！")
        last_current_year = (get_year() - 1) * 1000
        max_times = 160
        while len(data) != 3:
            data = spider(last_current_year + max_times, get_current_number(), "predict")[[x[0] for x in BOLL_NAME]]
            time.sleep(np.random.random(1).tolist()[0])
            max_times -= 1
    # 预测红球
    with red_graph.as_default():
        reverse_sequence = tf.compat.v1.get_default_graph().get_tensor_by_name(pred_key_d[BOLL_NAME[0][0]])
        red_pred = red_sess.run(reverse_sequence, feed_dict={
            "red_inputs:0": red_data.reshape(batch_size, windows_size, sequence_len),
            "sequence_length:0": np.array([sequence_len] * 1)
        })
    # 预测蓝球
    with blue_graph.as_default():
        softmax = tf.compat.v1.get_default_graph().get_tensor_by_name(pred_key_d[BOLL_NAME[1][0]])
        blue_pred = blue_sess.run(softmax, feed_dict={
            "blue_inputs:0": blue_data.reshape(batch_size, windows_size)
        })
    # 拼接结果
    ball_name_list = ["{}_{}".format(name[1], i) for name, i in red_name_list] + [BOLL_NAME[1][1]]
    pred_result_list = red_pred[0].tolist() + blue_pred.tolist()
    return json.dumps(
        {b_name: int(res) + 1 for b_name, res in zip(ball_name_list, pred_result_list)}
    ).encode('utf-8').decode('unicode_escape')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
