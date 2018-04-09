# **卷积神经网络**

# 一、摘要

​	卷积网络（Convolutional network）也叫神经网络，是一种专门用来处理具有类似网格结构的数据的神经网络。例如时间序列数据和图像数据（可以看做二维的像素网络）。卷积网络在诸多应用领域表现得都比较出色。**卷积网络是指那些至少在网络的一层中使用卷积运算来代替 一般的矩阵乘法运算的神经网络。**

# 二、卷积运算

​	在通常形式中，卷积是两个实变函数的一种数学运算。对于卷积连续的表达式我们可以表示为：

​				$$ s(t) = \int  \mathrm { x(a)}\mathrm{w(t-a)}\mathrm { d } x$$

而离散的表达式为：

​				$$s(t) = (x*w)(t) = \displaystyle\sum_{a=-\infty}^{\infty} x(a)w(t-a)$$

在卷积网络中，卷积的第一个参数(函数x)通常叫做输入，第二个参数（函数w）叫做卷积函数。输出有时被称作为特征映射。

​	在机器学习中，输入通常是多维数组的数据，而核通常是由学习优化得到的多维数组的参数。所以，在卷积层的网络中通常是一次在多个 维度上进行卷积运算。例如，如果把一张二维的图像I作为输入，这时我们要使用的核函数也是一个二维的核K：

​				$$S(i,j) = (I*K)(i,j) = \sum \sum I(m,n)K(i-m,j-n)$$

对于卷积的运算时可以交换的，所以：

​				$$S(i,j) = (I*K)(i,j) = \sum \sum I(i-m,j-n)K(m,n)$$

而在机器学习的库中，通常用到的m与n通常都比较小。

# 三、神经网络与卷积神经网络

 	卷积神经网络与普通神经网络的区别在于，卷积神经网络包含了一个由卷积层和子采样层构成的特征抽取器，而且在普通神经网络中，由于数据量很大，所以在隐层中的神经元一般也会比较多，所以使用全连接的方式会使他们的W参数量很大，容易过拟合。卷积神经网络解决的问题是参数量大的问题，在卷积神经网络的卷积层中，一个神经元只与部分邻层神经元连接。在CNN的一个卷积层中，通常包含若干个特征平面(featureMap)，每个特征平面由一些矩形排列的的神经元组成，同一特征平面的神经元共享权值，这里共享的权值就是卷积核。卷积核一般以随机小数矩阵的形式初始化，在网络的训练过程中卷积核将学习得到合理的权值。共享权值（卷积核）带来的直接好处是减少网络各层之间的连接，同时又降低了过拟合的风险。

## 1.层级结构

### 1.1 普通神经网络的层级结构：

![](https://raw.githubusercontent.com/zhi-z/DeepLearning/master/Convolutional_Neural_Networks/image/DNNstructure.png)

### 1.2 卷积网络层级结构：

​	有与普通神经网络的层级结构，不同层之间有不同的运算与功能。

![1523152503000](https://raw.githubusercontent.com/zhi-z/DeepLearning/master/Convolutional_Neural_Networks/image/1.structure.png)

主要的层次有：数据输入成（Input layer)，卷积计算层（CONV layer)，激励层（Activation layer）,池化层（Pooling layer）,全连接层（FC layer）。

# 四、卷积神经网络

## 1.数据输入层(Input layer)

​	在数据输入层中，接收数据的输入，并处理。有三种常见的处理方式。

### 1.1 去均值

​	把输入的数据都中心化到坐标轴的中心位置。

​           ![原始图像](https://raw.githubusercontent.com/zhi-z/DeepLearning/master/Convolutional_Neural_Networks/image/original%20data.png)          ![](https://raw.githubusercontent.com/zhi-z/DeepLearning/master/Convolutional_Neural_Networks/image/zero-centered%20data.png)

### 1.2 归一化

​	把幅度归一到同样的范围，但对于图像的数据，一般数值在0到255，所以对于图像的数据，可以不做归一化。

​            ![](https://raw.githubusercontent.com/zhi-z/DeepLearning/master/Convolutional_Neural_Networks/image/original%20data.png)        ![](https://raw.githubusercontent.com/zhi-z/DeepLearning/master/Convolutional_Neural_Networks/image/normalized%20data.png)

### 1.3 PCA/白化

​	对于数据维度太高，但有些数据对结果不太相关的，可以通过PCA进行降维。白化是对数据每个特征轴上的幅度归一化。

![](https://raw.githubusercontent.com/zhi-z/DeepLearning/master/Convolutional_Neural_Networks/image/pca.png)

## 2.卷积计算层（CONV  layer）

​	对于卷积层的参数是有一组可学习的滤波器组成，滤波器在空间上都很小。卷积层中每个神经元在空间上的连接的输入是局部的，但是他会连接到所有的深度（整张图片），每一个神经元都有一个W，对图像处理可以理解为滤波这组W通过滤波器对图像做变量，拿到该神经元觉得有用的信息。窗口滑动，对局部数据进行计算。

​	对于卷积计算的过程中会涉及到三个概念：深度（depth）、步长（stride）、零填充（zero-padding）。对于深度，我们可以理解为我们想要使用的滤波器（神经元）数量，每个滤波器的学习在输入中寻找不同的东西，对于他认为有用的特征就会抽取。对于幅度，当步幅为1时，我们一次将滤镜移动一个像素，幅度为二的时候移动两个像素。零填充，零填充的优点在于，它可以让我们控制输出体积的空间大小（最常见的是，我们很快就会看到，使用它来精确地保留输入体积的空间大小，以便输入和输出宽度和高度是相同的）。

![](https://raw.githubusercontent.com/zhi-z/DeepLearning/master/Convolutional_Neural_Networks/image/depthcol.jpg)

​	红色的示例输入体积（例如32x32x3 CIFAR-10图像）以及第一个卷积层中的示例体积的神经元。卷积层中的每个神经元仅在空间上连接到输入体积中的局部区域，但是连接到全深度（即所有颜色通道）。请注意，沿深度有多个神经元（本例中为5个），对于计算过程的动图如下，能够更清楚的看到计算的基本过程，但这是在一个图片一个通道上，一个神经元的计算：

![](C:\Users\JH\Documents\GitHub\DeepLearning\Convolutional_Neural_Networks\image\20170421182850369.gif)

下面举个栗子，以图片的形式展示卷积的计算过程：

原始图片的与特定的神经元(filter)做卷积运算，两个3x3的矩阵作相乘后再相加，以下图为例，计算：$0*0 + 0*0 + 0*1+ 0*1 + 1*0 + 0*0 + 0*0 + 0*1 + 0*1 =0$

![](https://raw.githubusercontent.com/zhi-z/DeepLearning/master/Convolutional_Neural_Networks/image/1_CO0yrGvAE7jw6JfGqCMRPg.png)

依序遍历完整张图片可得：

![](https://raw.githubusercontent.com/zhi-z/DeepLearning/master/Convolutional_Neural_Networks/image/1_Klv6ebMkjVmAEP4XkMxTXQ.png)

中间的神经元（Feature Detector）会随机产生好几种抽取图片特征的方式，他会帮助我们提取屠屏中的特征，就像人的大脑在判断这个图片是什么东西也是根据形状来推测，

![](https://raw.githubusercontent.com/zhi-z/DeepLearning/master/Convolutional_Neural_Networks/image/1_AJeWQ88UnmfkJ4_sFOT-YA.png)

以上就是卷积层的计算过程，通过卷积层抽取图片的特征。











