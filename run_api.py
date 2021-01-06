# -*- coding:utf-8 -*-
"""
Author: BigCat
"""
import json
import time
import datetime
import numpy as np
from config import *
from flask import Flask
from get_train_data import get_current_number, spider, pd
from tensorflow.keras.models import load_model

# load model
model_red_1 = load_model('model/lstm_model_红球号码_1.h5')
model_red_2 = load_model('model/lstm_model_红球号码_2.h5')
model_red_3 = load_model('model/lstm_model_红球号码_3.h5')
model_red_4 = load_model('model/lstm_model_红球号码_4.h5')
model_red_5 = load_model('model/lstm_model_红球号码_5.h5')
model_red_6 = load_model('model/lstm_model_红球号码_6.h5')
model_blue = load_model('model/lstm_model_蓝球.h5')
model_list = [
    model_red_1, model_red_2, model_red_3, model_red_4, model_red_5, model_red_6, model_blue
]
print("[INFO] 模型加载成功！")

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
    data = spider(str(int(get_current_number()) - diff_number), get_current_number(), "predict")[BOLL_NAME]
    if len(data) != 3:
        print("[WARN] 期号出现跳期，期号不连续！开始查找最近上一期期号！本期预测时间较久！")
        last_current_year = (get_year() - 1) * 1000
        max_times = 160
        while len(data) != 3:
            data = spider(last_current_year + max_times, get_current_number(), "predict")[BOLL_NAME]
            time.sleep(np.random.random(1).tolist()[0])
            max_times -= 1
    result = []
    for i, model in enumerate(model_list):
        boll_name = BOLL_NAME[i]
        data_list = [int(x) for x in data[boll_name].tolist()]
        p_data = np.array(data_list).reshape([-1, windows_size, 1]).astype(np.float32)
        result.extend(model.predict_classes(p_data))
    return json.dumps(
        {b_name: int(res) + 1 for b_name, res in zip(BOLL_NAME, result)}
    ).encode('utf-8').decode('unicode_escape')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
