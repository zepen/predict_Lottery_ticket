# -*- coding:utf-8 -*-
"""
Author: BigCat
"""
import json
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import config

# 关闭eager模式
tf.compat.v1.disable_eager_execution()

red_graph = tf.compat.v1.Graph()
with red_graph.as_default():
    red_saver = tf.compat.v1.train.import_meta_graph("{}red_ball_model.ckpt.meta".format(config.red_ball_model_path))
red_sess = tf.compat.v1.Session(graph=red_graph)
red_saver.restore(red_sess, "{}red_ball_model.ckpt".format(config.red_ball_model_path))
print("[INFO] 已加载红球模型！")

blue_graph = tf.compat.v1.Graph()
with blue_graph.as_default():
    blue_saver = tf.compat.v1.train.import_meta_graph("{}blue_ball_model.ckpt.meta".format(config.blue_ball_model_path))
blue_sess = tf.compat.v1.Session(graph=blue_graph)
blue_saver.restore(blue_sess, "{}blue_ball_model.ckpt".format(config.blue_ball_model_path))
print("[INFO] 已加载蓝球模型！")

# 加载关键节点名
with open("{}{}".format(config.model_path, config.pred_key_name)) as f:
    pred_key_d = json.load(f)


def get_year():
    """截取年份
    eg：2020-->20, 2021-->21
    :return:
    """
    return int(str(datetime.datetime.now().year)[-2:])


def get_predict_result():
    with open('{}{}'.format(config.train_data_path, config.train_data_file), 'r', encoding='utf-8') as f:
        data_raw = json.load(f)
    diff_number = config.windows_size
    data = pd.DataFrame(data_raw[:diff_number])
    red_name_list = ["{}号码_{}".format(config.BOLL_NAME[0], i + 1) for i in range(config.sequence_len)]
    red_data = data[red_name_list].values.astype(int) - 1
    blue_data = data[[config.BOLL_NAME[1]]].values.astype(int) - 1
    # 预测红球
    with red_graph.as_default():
        reverse_sequence = tf.compat.v1.get_default_graph().get_tensor_by_name(pred_key_d[config.BOLL_NAME[0]])
        red_pred = red_sess.run(
            reverse_sequence,
            feed_dict={
                "red_inputs:0": red_data.reshape(config.batch_size, config.windows_size, config.sequence_len),
                "sequence_length:0": np.array([config.sequence_len] * 1),
            },
        )
    # 预测蓝球
    with blue_graph.as_default():
        softmax = tf.compat.v1.get_default_graph().get_tensor_by_name(pred_key_d[config.BOLL_NAME[1]])
        blue_pred = blue_sess.run(
            softmax, feed_dict={"blue_inputs:0": blue_data.reshape(config.batch_size, config.windows_size)}
        )
    # 拼接结果
    ball_name_list = red_name_list + [config.BOLL_NAME[1]]
    pred_result_list = red_pred[0].tolist() + blue_pred.tolist()
    return (
        json.dumps({b_name: int(res) + 1 for b_name, res in zip(ball_name_list, pred_result_list)})
        .encode('utf-8')
        .decode('unicode_escape')
    )


if __name__ == '__main__':
    result = get_predict_result()
    print(result)
