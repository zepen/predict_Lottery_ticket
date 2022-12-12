# -*- coding: utf-8 -*-
"""
Author: BigCat
Modifier: KittenCN
"""
import os

ball_name = [
    ("红球", "red"),
    ("蓝球", "blue")
]

data_file_name = "data.csv"

name_path = {
    "ssq": {
        "name": "双色球",
        "path": "data/ssq/"
    },
    "dlt": {
        "name": "大乐透",
        "path": "data/dlt/"
    },
    "qxc": {
        "name": "七星彩",
        "path": "data/qxc/"
    },
    "pls": {
        "name": "排列三",
        "path": "data/pls/"
    },
    "kl8": {
        "name": "快乐8",
        "path": "data/kl8/"
    },
}

model_path = os.getcwd() + "/model/"

model_args = {
    "kl8": {
        "model_args": {
            "windows_size": 3,
            "batch_size": 1,
            "red_sequence_len": 20,
            "sequence_len": 20,
            "red_n_class": 80,
            "red_epochs": 1,
            "red_embedding_size": 20,
            "red_hidden_size": 20,
            "red_layer_size": 1,
            "blue_sequence_len": 1,
            "blue_n_class": 0,
            "blue_epochs": 0,
            "blue_embedding_size": 0,
            "blue_hidden_size": 0,
            "blue_layer_size": 0,
        },
        "train_args": {
            "red_learning_rate": 0.001,
            "red_beta1": 0.9,
            "red_beta2": 0.999,
            "red_epsilon": 1e-08,
            "blue_learning_rate": 0.001,
            "blue_beta1": 0.9,
            "blue_beta2": 0.999,
            "blue_epsilon": 1e-08
        },
        "path": {
            "red": model_path + "/kl8/red_ball_model/",
            "blue": model_path + "/kl8/blue_ball_model/"
        },
        "pathname": {
            "name":"/kl8/"
        },
        "subpath": {
            "red": "/red_ball_model/",
            "blue": "/blue_ball_model/"
        }
    },
    "pls": {
        "model_args": {
            "windows_size": 3,
            "batch_size": 1,
            "red_sequence_len": 3,
            "sequence_len": 3,
            "red_n_class": 10,
            "red_epochs": 1,
            "red_embedding_size": 10,
            "red_hidden_size": 10,
            "red_layer_size": 1,
            "blue_sequence_len": 1,
            "blue_n_class": 0,
            "blue_epochs": 0,
            "blue_embedding_size": 0,
            "blue_hidden_size": 0,
            "blue_layer_size": 0,
        },
        "train_args": {
            "red_learning_rate": 0.001,
            "red_beta1": 0.9,
            "red_beta2": 0.999,
            "red_epsilon": 1e-08,
            "blue_learning_rate": 0.001,
            "blue_beta1": 0.9,
            "blue_beta2": 0.999,
            "blue_epsilon": 1e-08
        },
        "path": {
            "red": model_path + "/pls/red_ball_model/",
            "blue": model_path + "/pls/blue_ball_model/"
        },
        "pathname": {
            "name":"/pls/"
        },
        "subpath": {
            "red": "/red_ball_model/",
            "blue": "/blue_ball_model/"
        }
    },
    "ssq": {
        "model_args": {
            "windows_size": 3,
            "batch_size": 1,
            "red_sequence_len": 6,
            "sequence_len": 6,
            "red_n_class": 33,
            "red_epochs": 1,
            "red_embedding_size": 32,
            "red_hidden_size": 32,
            "red_layer_size": 1,
            "blue_sequence_len": 1,
            "blue_n_class": 16,
            "blue_epochs": 1,
            "blue_embedding_size": 32,
            "blue_hidden_size": 32,
            "blue_layer_size": 1,
        },
        "train_args": {
            "red_learning_rate": 0.001,
            "red_beta1": 0.9,
            "red_beta2": 0.999,
            "red_epsilon": 1e-08,
            "blue_learning_rate": 0.001,
            "blue_beta1": 0.9,
            "blue_beta2": 0.999,
            "blue_epsilon": 1e-08
        },
        "path": {
            "red": model_path + "/ssq/red_ball_model/",
            "blue": model_path + "/ssq/blue_ball_model/"
        },
        "pathname": {
            "name":"/ssq/"
        },
        "subpath": {
            "red": "/red_ball_model/",
            "blue": "/blue_ball_model/"
        }
    },
    "dlt": {
        "model_args": {
            "windows_size": 3,
            "batch_size": 1,
            "red_sequence_len": 5,
            "red_n_class": 35,
            "red_epochs": 1,
            "red_embedding_size": 32,
            "red_hidden_size": 32,
            "red_layer_size": 1,
            "blue_sequence_len": 2,
            "blue_n_class": 12,
            "blue_epochs": 1,
            "blue_embedding_size": 32,
            "blue_hidden_size": 32,
            "blue_layer_size": 1
        },
        "train_args": {
            "red_learning_rate": 0.001,
            "red_beta1": 0.9,
            "red_beta2": 0.999,
            "red_epsilon": 1e-08,
            "blue_learning_rate": 0.001,
            "blue_beta1": 0.9,
            "blue_beta2": 0.999,
            "blue_epsilon": 1e-08
        },
        "path": {
            "red": model_path + "/dlt/red_ball_model/",
            "blue": model_path + "/dlt/blue_ball_model/"
        },
        "pathname": {
            "name":"/dlt/"
        },
        "subpath": {
            "red": "/red_ball_model/",
            "blue": "/blue_ball_model/"
        }
    }
}

# 模型名
pred_key_name = "key_name.json"
red_ball_model_name = "red_ball_model"
blue_ball_model_name = "blue_ball_model"
extension = "ckpt"
