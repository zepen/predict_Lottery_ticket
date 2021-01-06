# -*- coding: utf-8 -*-
"""
Author: BigCat
"""

URL = "https://datachart.500.com/ssq/history/"
path = "newinc/history.php?start={}&end="

BOLL_NAME = [
    "红球号码_1",
    "红球号码_2",
    "红球号码_3",
    "红球号码_4",
    "红球号码_5",
    "红球号码_6",
    "蓝球"
]

train_data_path = "data/"
train_data_file = "data.csv"

# 预测序列
windows_size = 3
