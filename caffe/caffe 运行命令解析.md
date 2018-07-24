# caffe 运行命令解析

## 1 简介

caffe的运行提供三种接口：c++接口（命令行）、python接口和matlab接口。本文先对命令行进行解析，后续会依次介绍其它两个接口。

caffe的c++主程序（caffe.cpp)放在根目录下的tools文件夹内, 当然还有一些其它的功能文件，如：convert_imageset.cpp, train_net.cpp, test_net.cpp等也放在这个文件夹内。经过编译后，这些文件都被编译成了可执行文件，放在了 ./build/tools/ 文件夹内。因此我们要执行caffe程序，都需要加 ./build/tools/ 前缀。

## 2 命令解析

### 2.1 train命令解析

直接对命令进行解析。如：

```
sudo sh ./build/tools/caffe train --solver=examples/mnist/train_lenet.sh
```

caffe 程序运行的格式如下：

```
caffe <command> <args>
```

其中<command> 可选的参数有：

- train：训练或者finetune模型（model）
- test：测试模型
- device_query:显示GPU信息
- time：显示程序执行时间

**<args>**可选的参数有：

* -solver

  必选参数，一个protocol buffer类型的文件，即模型的配置文件。如：

  ```
  ./build/tools/caffe train -solver examples/mnist/lenet_solver.prototxt
  ```

* -gpu

  可选参数，该参数用来制定用哪一块gpu运行，根据gpu的id进行选择，如果设置为'-gpu all'则使用所有的gpu运行。如使用第二块gpu运行。如：

  ```
  ./build/tools/caffe train -solver examples/mnist/lenet_solver.prototxt -gpu 2
  ```

* -snapshot

  可选参数。该参数用来从快照（snapshot)中恢复训练。可以在solver配置文件设置快照，保存solverstate。如：

  ```
  ./build/tools/caffe train -solver examples/mnist/lenet_solver.prototxt -snapshot examples/mnist/lenet_iter_5000.solverstate
  ```

* -weights

  可选参数。用预先训练好的权重来fine-tuning模型，需要一个caffemodel，不能和-snapshot同时使用。如：

  ```
  ./build/tools/caffe train -solver examples/finetuning_on_flickr_style/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
  ```

* -iteration

  可选参数，迭代次数，默认为50。 如果在配置文件文件中没有设定迭代次数，则默认迭代50次。

* -model

  可选参数，定义在protocol buffer文件中的模型。也可以在solver配置文件中指定。

* -sighup_effect

  可选参数。用来设定当程序发生挂起事件时，执行的操作，可以设置为snapshot, stop或none, 默认为snapshot。

* -sigint_effect

可选参数。用来设定当程序发生键盘中止事件时（ctrl+c), 执行的操作，可以设置为snapshot, stop或none, 默认为stop。

### 2.2 test命令解析

与train很多是一样的，test参数用在测试阶段，最终用于结果输出，可以设定需要输出的accuracy和loss，假设我们在验证集中验证已经训练好的模型，可以书写成以下方式：

```
./build/tools/caffe test -model examples/mnist/lenet_train_test.prototxt -weights examples/mnist/lenet_iter_10000.caffemodel -gpu 0 -iterations 100
```

这个例子比较长，不仅用到了test参数，还用到了-model, -weights,  -gpu和-iteration四个参数。意思是利用训练好了的权重（-weight)，输入到测试模型中(-model)，用编号为0的gpu(-gpu)测试100次(-iteration)。

### 2.3 time命令解析

**time**参数用来在屏幕上显示程序运行时间。如：

```
./build/tools/caffe time -model examples/mnist/lenet_train_test.prototxt -iterations 10
```

这个例子用来在屏幕上显示lenet模型迭代10次所使用的时间。包括每次迭代的forward和backward所用的时间，也包括每层forward和backward所用的平均时间。

```
 ./build/tools/caffe time -model examples/mnist/lenet_train_test.prototxt -gpu 0
```

这个例子用来在屏幕上显示lenet模型用gpu迭代50次所使用的时间。

```
./build/tools/caffe time -model examples/mnist/lenet_train_test.prototxt -weights examples/mnist/lenet_iter_10000.caffemodel -gpu 0 -iterations 10
```

利用给定的权重，利用第一块gpu，迭代10次lenet模型所用的时间。

### 2.4 device_quer命令解析

**device_query**参数用来诊断gpu信息。

```
./build/tools/caffe train -solver examples/mnist/lenet_solver.prototxt -gpu 0,1
```

多个gpu运行：

```
./build/tools/caffe train -solver examples/mnist/lenet_solver.prototxt -gpu 0,1
```

使用GPU0 和GPU1对网络进行训练。

```
./build/tools/caffe train -solver examples/mnist/lenet_solver.prototxt -gpu all
```

使用计算机所用的GPU对网络进行训练。



## 参考文献

[Caffe学习系列(10)：命令行解析](https://www.cnblogs.com/denny402/p/5076285.html)

