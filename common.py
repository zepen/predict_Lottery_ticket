# -*- coding:utf-8 -*-
"""
Author: KittenCN
"""
import requests
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from config import *

def get_url(name):
    """
    :param name: 玩法名称
    :return:
    """
    url = "https://datachart.500.com/{}/history/".format(name)
    path = "newinc/history.php?start={}&end={}&limit={}"
    if name == "qxc" or name == "pls":
        path = "inc/history.php?start={}&end={}&limit={}"
    return url, path

def get_current_number(name):
    """ 获取最新一期数字
    :return: int
    """
    url, _ = get_url(name)
    if name in ["qxc", "pls"]:
        r = requests.get("{}{}".format(url, "inc/history.php"), verify=False)
    elif name in ["ssq", "dlt"]:
        r = requests.get("{}{}".format(url, "history.shtml"), verify=False)
    r.encoding = "gb2312"
    soup = BeautifulSoup(r.text, "lxml")
    current_num = soup.find("div", class_="wrap_datachart").find("input", id="end")["value"]
    return current_num


def spider(name="ssq", start=1, end=999999, mode="train"):
    """ 爬取历史数据
    :param name 玩法
    :param start 开始一期
    :param end 最近一期
    :param mode 模式，train：训练模式，predict：预测模式（训练模式会保持文件）
    :return:
    """
    if mode == "train":
        url, path = get_url(name)
        limit = int(end) - int(start) + 1
        url = "{}{}".format(url, path.format(int(start), int(end), limit))
        r = requests.get(url=url, verify=False)
        r.encoding = "gb2312"
        soup = BeautifulSoup(r.text, "lxml")
        if name in ["ssq", "dlt"]:
            trs = soup.find("tbody", attrs={"id": "tdata"}).find_all("tr")
        elif name in ["qxc", "pls"]:
            trs = soup.find("div", class_="wrap_datachart").find("table", id="tablelist").find_all("tr")
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
            elif name == "pls":
                if tr.find_all("td")[0].get_text().strip() == "注数" or tr.find_all("td")[1].get_text().strip() == "中奖号码":
                    continue
                item[u"期数"] = tr.find_all("td")[0].get_text().strip()
                # if tr.find_all("td")[1].get_text().strip() == "中奖号码":
                #     for i in range(3):
                #         item[u"红球_{}".format(i+1)] = "红球号码{}".format(i+1)
                # else:
                numlist = tr.find_all("td")[1].get_text().strip().split(" ")
                for i in range(3):
                    item[u"红球_{}".format(i+1)] = numlist[i]
                data.append(item)
            else:
                logger.warning("抱歉，没有找到数据源！")

        df = pd.DataFrame(data)
        df.to_csv("{}{}".format(name_path[name]["path"], data_file_name), encoding="utf-8")
        return pd.DataFrame(data)

    elif mode == "predict":
        ori_data = pd.read_csv("{}{}".format(name_path[name]["path"], data_file_name))
        data = []
        for i in range(len(ori_data)):
            item = dict()
            if ori_data.iloc[i, 1] < int(start) or ori_data.iloc[i, 1] > int(end):
                continue
            if name == "ssq":
                item[u"期数"] = ori_data.iloc[i, 1]
                for j in range(6):
                    item[u"红球_{}".format(j+1)] = ori_data.iloc[i, j+2]
                item[u"蓝球"] = ori_data.iloc[i, 8]
                data.append(item)
            elif name == "dlt":
                item[u"期数"] = ori_data.iloc[i, 1]
                for j in range(5):
                    item[u"红球_{}".format(j+1)] = ori_data.iloc[i, j+2]
                for k in range(2):
                    item[u"蓝球_{}".format(k+1)] = ori_data.iloc[i, 7+k]
                data.append(item)
            elif name == "pls":
                item[u"期数"] = ori_data.iloc[i, 1]
                for j in range(3):
                    item[u"红球_{}".format(j+1)] = ori_data.iloc[i, j+2]
                data.append(item)
            else:
                logger.warning("抱歉，没有找到数据源！")
        return pd.DataFrame(data)