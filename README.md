# 彩票AI预测（目前支持双色球，大乐透，排列三）

核心代码 fork 自：https://github.com/zepen/predict_Lottery_ticket
并按照我的思路进行了增强。

目前的代码已经同时支持CPU和GPU计算，但是估计是网络比较浅，GPU速度反而不如CPU。目前我在考虑修改网络结构，并迁移到我比较熟悉的pytorch框架之下.

## Installing
        
* step1，安装anaconda(可参考https://zhuanlan.zhihu.com/p/32925500)；

* step2，创建一个conda环境，conda create -n your_env_name python=3.6；
       
* step3，进入创建conda的环境 conda activate your_env_name，然后根据自己机器的状况，选择CPU或者GPU模式，并在requirement文件中，把对应版本的Tensorflow解除注释，并执行pip install -r requirements.txt；如果不确定哪个版本更合适，建议使用cpu版本
* 备注：根据我个人的测试，不推荐使用其他版本的tensorflow,如果因为硬件原因，一定要用更高或者更低版本的tensorflow,请同时更新tensorflow-addons，pandas，numpy的版本。
       
* step4，按照Getting Started执行即可

## Getting Started

```python
python get_data.py  --name ssq  # 执行获取双色球训练数据
```
如果出现解析错误，应该看看网页 http://datachart.500.com/ssq/history/newinc/history.php 是否可以正常访问
若要大乐透，替换参数 --name dlt 即可

```python
python run_train_model.py --name ssq  --windows_size 3,5,7 --red_epochs 1 --blue_epochs 1 --batch_size 1  # 执行训练双色球模型
``` 
开始模型训练，先训练红球模型，再训练蓝球模型，模型参数和超参数在 config.py 文件中自行配置
具体训练时间消耗与模型参数和超参数相关。
若要多个窗口尺寸依次训练，替换参数 --windows_size 3,5,7 即可
red_epochs 为红球训练次数
blue_epochs 为篮球训练次数
batch_size 为每轮训练的数量

```python
python run_predict.py  --name ssq --windows_size 3,5,7  # 执行双色球模型预测
```
预测结果会打印在控制台

## Update

* 有盆友反馈想要个大乐透的预测玩法，加入对大乐透的数据爬取，模型训练，模型预测等功能，通过传入执行参数 --name dlt即可。

* 为了降低本项目的使用门槛，废弃docker模式和微服务，按照Getting Started执行脚本，即可获取预测结果。

* 非常开心有更多的同志们关注项目，并且提出了很多宝贵的问题，但是由于工作较忙，没有给大家比较完善的解答，再次说句抱歉，
大部分问题都是安装依赖问题，我更新了requirements.txt中相关库版本，应该可以解决。

* 之前有issue反应，因为不同红球模型预测会有重复号码出现，所以将红球序列整体作为一个序列模型看待，推翻之前红球之间相互独立设定，
因为序列模型预测要引入crf层，相关API必须在 tf.compat.v1.disable_eager_execution()下，故整个模型采用 1.x 构建和训练模式，
在 2.x 的tensorflow中 tf.compat.v1.XXX 保留了 1.x 的接口方式。
