# 双色球+大乐透彩票AI预测

有问题，请联系客服（客服1群：246714623（满员请加2群），客服2群：980203303）

公众号

![avatar](img/gzh.png)


## Installing
        
* step1，安装anaconda(可参考https://zhuanlan.zhihu.com/p/32925500)；

* step2，创建一个conda环境，conda create -n your_env_name python=3.6；
       
* step3，进入创建conda的环境 conda activate your_env_name，然后执行pip install -r requirements.txt；
       
* step4，按照Getting Started执行即可，推荐使用PyCharm

## Getting Started

```python
python get_data.py  --name ssq  # 执行获取双色球训练数据
```
如果出现解析错误，应该看看网页 http://datachart.500.com/ssq/history/newinc/history.php 是否可以正常访问
若要大乐透，替换参数 --name dlt 即可

```python
python run_train_model.py --name ssq  # 执行训练双色球模型
``` 
开始模型训练，先训练红球模型，再训练蓝球模型，模型参数和超参数在 config.py 文件中自行配置
具体训练时间消耗与模型参数和超参数相关。

```python
python run_predict.py  --name ssq # 执行双色球模型预测
```
预测结果会打印在控制台

## Update

* 新增模型预测评估，可以自行调整训练集和测试集比例，建议训练集采样比例高于0.5

* 修复大乐透蓝球号码预测超出取值范围问题，修复训练传参数导致数据维度不匹配问题

* 有盆友反馈想要个大乐透的预测玩法，加入对大乐透的数据爬取，模型训练，模型预测等功能，通过传入执行参数 --name dlt即可。

* 为了降低本项目的使用门槛，废弃docker模式和微服务，按照Getting Started执行脚本，即可获取预测结果。

* 非常开心有更多的同志们关注项目，并且提出了很多宝贵的问题，但是由于工作较忙，没有给大家比较完善的解答，再次说句抱歉，
大部分问题都是安装依赖问题，我更新了requirements.txt中相关库版本，应该可以解决。

* 之前有issue反应，因为不同红球模型预测会有重复号码出现，所以将红球序列整体作为一个序列模型看待，推翻之前红球之间相互独立设定，
因为序列模型预测要引入crf层，相关API必须在 tf.compat.v1.disable_eager_execution()下，故整个模型采用 1.x 构建和训练模式，
在 2.x 的tensorflow中 tf.compat.v1.XXX 保留了 1.x 的接口方式。
