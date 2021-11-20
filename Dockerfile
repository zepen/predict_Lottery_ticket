FROM python:3.6.12

# 代码添加到code文件夹
ADD . /usr/src/app

# 设置app文件夹是工作目录
WORKDIR /usr/src/app

# 安装支持
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "get_train_data.py"]
CMD [ "python", "train_model.py"]
CMD [ "gunicorn", "-c gunicorn_conf.py", "run_api:app"]