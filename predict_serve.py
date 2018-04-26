# -*- coding:utf-8 -*-
"""
Author: Niuzepeng
"""
from flask import request, Flask
from keras.models import load_model

# load model
for x in range(7):
    exec("model_" + str(x) + ' = load_model("model/model_" + str(x) + ".h5")')

app = Flask(__name__)


@app.route('/number', methods=['POST'])
def obtain_predict_result():
    try:
        if request.method == "POST":
            if len(request.args) != 0:
                pass
    except Exception as e:
        print(e)

if __name__ == '__main__':
    pass
