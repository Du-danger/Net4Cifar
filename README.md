# Net4Cifar: A simple net for cifar dataset classfication.

## 1. Dataset Description
CIFAR-10 是由 Hinton 的学生 Alex Krizhevsky 和 Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含 10 个类别的 RGB 彩色图片：飞机（airplane）、汽车（automobile）、鸟类（bird）、猫（cat）、鹿（deer）、狗（dog）、蛙类（frog）、马（horse）、船（ship）和卡车（truck）。图片的尺寸为: 32×32，数据集中一共有50000 张训练图片以及10000 张测试图片。
## 2. Method
### 2.1 数据处理
对数据集中的图像进行数据增强和标准化处理：随机裁剪，随机反转，转为tensor，标准化
### 2.2 模型构建
以 net10 = [32, 'M', 64, 'M', 128, 'M', 256, 'M', 512, 'M'] 为例。

① 特征提取层：一共包含十个卷积层，其中每个cov2d层（32，64，...，512）后都跟有一个最大池化层'M'，，最后针对每个图像输出的是一个512维的向量。

② 分类层：对512维经过一个线性层，转化为10维的向量，每一维对应着属于每个类别的概率。
### 2.3 损失设计
损失设计，利用交叉熵损失函数来判定实际的输出与期望的输出的接近程度。
![image](https://user-images.githubusercontent.com/73435352/194759845-aa621a6e-2ce4-4816-a3de-c8d082321c3c.png)

## 3. Hyper-parameter Tuning
这里选用模型的深度作为一个超参数。

① net10: 32, 'M', 64, 'M', 128, 'M', 256, 'M', 512, 'M'

② net15: 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'

③ net20: 32, 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'

最终的结果如下：
| netX | Acc |
|------|-----|
| net10|     |
| net15|     |
| net20|     |
