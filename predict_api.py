# -*- coding:utf-8 -*-
"""
Author: Niuzepeng
"""
import json
import numpy as np
from config import *
from flask import Flask
from get_train_data import get_current_number, spider
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


@app.route('/')
def main():
    return "Welcome to use!"


@app.route('/predict_api', methods=['GET'])
def get_predict_result():
    data = spider(str(int(get_current_number()) - 2), get_current_number(), "predict")[BOLL_NAME]
    result = []
    for i, model in enumerate(model_list):
        boll_name = BOLL_NAME[i]
        data_list = [int(x) for x in data[boll_name].tolist()]
        p_data = np.array(data_list).reshape([-1, 3, 1]).astype(np.float32)
        result.extend(model.predict_classes(p_data))
    return json.dumps({b_name: int(res) + 1 for b_name, res in zip(BOLL_NAME, result)})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
