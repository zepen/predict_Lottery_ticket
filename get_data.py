# -*- coding:utf-8 -*-
"""
Author: BigCat
Modifier: KittenCN
"""
import argparse
from loguru import logger
from config import os, name_path
from common import *

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="ssq", type=str, help="选择爬取数据")
args = parser.parse_args()

def run(name):
    """
    :param name: 玩法名称
    :return:
    """
    current_number = get_current_number(name)
    logger.info("【{}】最新一期期号：{}".format(name_path[name]["name"], current_number))
    logger.info("正在获取【{}】数据。。。".format(name_path[name]["name"]))
    if not os.path.exists(name_path[name]["path"]):
        os.makedirs(name_path[name]["path"])
    data = spider(name, 1, current_number, "train")
    if "data" in os.listdir(os.getcwd()):
        logger.info("【{}】数据准备就绪，共{}期, 下一步可训练模型...".format(name_path[name]["name"], len(data)))
    else:
        logger.error("数据文件不存在！")


if __name__ == "__main__":
    if not args.name:
        raise Exception("玩法名称不能为空！")
    else:
        run(name=args.name)
