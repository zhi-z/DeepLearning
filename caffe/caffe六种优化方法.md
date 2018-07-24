# caffe六种优化方法

## 1 简介

所谓的优化方法是指对于训练网络的loss进行优化。caffe中在Solver配置，在神经网络中，用forward pass来求解loss，用backward pass来求解梯度。六种优化方法分别为。

- Stochastic Gradient Descent (type: "SGD"),
- AdaDelta (type: "AdaDelta")
- Adaptive Gradient (type: "AdaGrad")
- Adam (type: "Adam")
- Nesterov’s Accelerated Gradient (type: "Nesterov") 
- RMSprop (type: "RMSProp")

## 2 优化方法

### 2.1 随机梯度下降（SGD）

随机梯度下降法主要为了解决梯度计算，由于随机梯度下降法的引入，童话从哪个将梯度下降法分为三种类型：

- 批梯度下降法(GD)

  原始的梯度下降法

- 随机梯度下降法(SGD)

  每次梯度计算只使用一个样本

  避免在类似样本上计算梯度造成的冗余计算

  增加了跳出当前的局部最小值的潜力

  在逐渐缩小学习率的情况下,有与批梯度下降法类似的收敛速度

- 小批量随机梯度下降法(Mini Batch SGD)

  每次梯度计算使用一个小批量样本

  梯度计算比单样本更加稳定

  可以很好的利用现成的高度优化的矩阵运算工具

通常神经网络训练中把Mini Baatch SGD 称为SGD。



***未完待续***

