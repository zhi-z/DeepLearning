# Caffe 图片数据的均值

## 1 简介

 图片减去均值后，再进行训练和测试，会提高速度和精度。因此，一般在各种模型中都会有这个操作。

那么这个均值怎么来的呢，实际上就是计算所有训练样本的平均值，计算出来后，保存为一个均值文件，在以后的测试中，就可以直接使用这个均值来相减，而不需要对测试图片重新计算。

## 2 二进制格式均值计算

caffe中使用的均值数据格式是binaryproto,  作者为我们提供了一个计算均值的文件compute_image_mean.cpp，放在caffe根目录下的tools文件夹里面。编译后的可执行体放在 build/tools/ 下面，我们直接调用就可以了，命令格式：

```
# sudo build/tools/compute_image_mean examples/mnist/mnist_train_lmdb examples/mnist/mean.binaryproto
```

- 参数说明：

  第一个参数：examples/mnist/mnist_train_lmdb， 表示需要计算均值的数据，格式为lmdb的训练数据。

  第二个参数：examples/mnist/mean.binaryproto， 计算出来的结果保存文件。

## 3 Python 格式均值计算

如果我们要使用python接口，或者我们要进行特征可视化，可能就要用到python格式的均值文件了。首先，我们用lmdb格式的数据，计算出二进制格式的均值，然后，再转换成python格式的均值。

我们可以编写一个python脚本来实现：

```
import numpy as np
import sys,caffe

if len(sys.argv)!=3:
    print("Usage: python convert_mean.py mean.binaryproto mean.npy")
    sys.exit()

blob = caffe.proto.caffe_pb2.BlobProto()
bin_mean = open( sys.argv[1] , 'rb' ).read()
blob.ParseFromString(bin_mean)
arr = np.array( caffe.io.blobproto_to_array(blob) )
npy_mean = arr[0]
np.save( sys.argv[2] , npy_mean )
```

并将这个文件保存成convert_mean.py文件，之后运行。

```
# sudo python convert_mean.py mean.binaryproto mean.npy
```

其中的 mean.binaryproto 就是经过前面步骤计算出来的二进制均值。mean.npy就是我们需要的python格式的均值。



