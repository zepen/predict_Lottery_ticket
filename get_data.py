#-*- coding:utf-8 -*-

# Author:longjiang



import requests
from bs4 import BeautifulSoup
import pandas as pd


def spider():
    url="https://datachart.500.com/ssq/history/newinc/history.php?start=1&end=18046"

    r=requests.get(url=url)

    r.encoding="gb2312"

    soup=BeautifulSoup(r.text,"lxml")

    trs= soup.find("tbody",attrs={"id":"tdata"}).find_all("tr")
    data=list()
    for tr in trs:
        item=dict()

        item[u"期数"]= tr.find_all("td")[0].get_text().strip()
        item[u"红球号码_1"] = tr.find_all("td")[1].get_text().strip()
        item[u"红球号码_2"] = tr.find_all("td")[2].get_text().strip()
        item[u"红球号码_3"] = tr.find_all("td")[3].get_text().strip()
        item[u"红球号码_4"] = tr.find_all("td")[4].get_text().strip()
        item[u"红球号码_5"] = tr.find_all("td")[5].get_text().strip()
        item[u"红球号码_6"] = tr.find_all("td")[6].get_text().strip()
        item[u"蓝球"] = tr.find_all("td")[7].get_text().strip()
        item[u"快乐星期天"] = tr.find_all("td")[8].get_text().strip()
        item[u"奖池奖金(元)"] = tr.find_all("td")[9].get_text().strip()
        item[u"一等奖_注数"] = tr.find_all("td")[10].get_text().strip()
        item[u"一等奖_奖金(元)"] = tr.find_all("td")[11].get_text().strip()
        item[u"二等奖_注数"] = tr.find_all("td")[12].get_text().strip()
        item[u"二等奖_奖金(元)"] = tr.find_all("td")[13].get_text().strip()
        item[u"总投注额(元)"] = tr.find_all("td")[14].get_text().strip()
        item[u"开奖日期"] = tr.find_all("td")[15].get_text().strip()

        data.append(item)


    df=pd.DataFrame(data)

    df.to_csv("data.csv",encoding="utf-8")


if __name__=="__main__":
    spider()