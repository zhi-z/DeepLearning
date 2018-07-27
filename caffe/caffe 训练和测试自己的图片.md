# caffe 训练和测试自己的图片

## 1. 数据准备

如果网络比较好，可以去[imagenet](http://www.image-net.org/download-images)下载。但是由于网络的原因我没有下载。这里测试的数据是在网上找的。总共有500张，分别为大巴车、恐龙、大象、鲜花和马，美个类有100张图片，分别以3、4、5、6、7开头，各为一类。需要的可以在这里下载。

之后为了训练作准备，我从每一类各取出20张作为测试集，其余的作为训练数据。那么，共有400张图作训练数据，测试数据为100张。在最后我把图片放到caffe根目录下的data文件夹下面，即训练图片目录为：data/re/train/，测试图片数据放在：data/re/test

## 2. 转换为lmdb格式

对于更详细的转换过程，可以参考我之气那的一篇博文，[图像数据转换成db（leveldb/lmdb）文件](https://blog.csdn.net/weixin_41863685/article/details/81227955).

### 2.1 生成图片list

- 首先，在examples下面创建一个myfile的文件夹，来用存放配置文件和脚本文件。然后编写一个脚本create_filelist.sh，用来生成train.txt和test.txt清单文件

```
# sudo mkdir examples/myfile
# sudo vi examples/myfile/create_filelist.sh
```

运行如上两行命令，然后把以下的脚本添加到create_filelist.sh.

```
#!/usr/bin/env sh
DATA=data/re/
Create_dir=examples/myfile

echo "Create train.txt..."
rm -rf $MY/train.txt
for i in 3 4 5 6 7 
do
find $DATA/train -name $i*.jpg | cut -d '/' -f4-5 | sed "s/$/ $i/">>$Create_dir/train.txt
done
echo "Create test.txt..."
rm -rf $MY/test.txt
for i in 3 4 5 6 7
do
find $DATA/test -name $i*.jpg | cut -d '/' -f4-5 | sed "s/$/ $i/">>$MY/test.txt
done
echo "All done"
```

运行脚本

```
# sudo sh examples/myfile/create_filelist.sh
```

结果：Terminal会输出，

```
Create train.txt...
Create test.txt...
All done
```

在文件夹中会看到以下文件，如图所示

<center>

![](https://raw.githubusercontent.com/zhi-z/DeepLearning/master/caffe/image/create_list_out.png)

</center>

并且在生成的train.txt文件中的内容如图所示。

<center>

![](https://raw.githubusercontent.com/zhi-z/DeepLearning/master/caffe/image/create_list_out_2.png)

</center>

到这里完成了把图片的目录添加到一个list中，相当与制作标签。

### 2.2 转换成lmdb格式

运行如下命令，进行脚本的编辑，这个脚本的作用就是就是把图片数据转换成lmdb数据。

```
# sudo vi examples/myfile/create_lmdb.sh
```

然后在插入如下脚本

```
#!/usr/bin/env sh

dir=examples/myfile

echo "Create train lmdb.."
rm -rf $dir/img_train_lmdb
build/tools/convert_imageset \
--shuffle \
--resize_height=256 \
--resize_width=256 \
/home/xxx/caffe/data/re/ \
$dir/train.txt \
$dir/img_train_lmdb

echo "Create test lmdb.."
rm -rf $dir/img_test_lmdb
build/tools/convert_imageset \
--shuffle \
--resize_width=256 \
--resize_height=256 \
/home/xxx/caffe/data/re/ \
$dir/test.txt \
$dir/img_test_lmdb

echo "All Done.."
```

注意，在保存输出的路径中的xxx要改为自己电脑的地址。因为图片大小不一，因此我统一转换成256*256大小。通过如下命令执行脚本

```
bash examples/myfile/create_lmdb.sh
```

输出的结果：

![](https://raw.githubusercontent.com/zhi-z/DeepLearning/master/caffe/image/caffe_re_out.png)

同时在 examples/myfile下面生成两个文件夹img_train_lmdb和img_test_lmdb，分别用于保存图片转换后的lmdb文件。如同所示。

![](https://raw.githubusercontent.com/zhi-z/DeepLearning/master/caffe/image/caffe_re_out_2.png)

## 3. 计算均值并保存

在深度学习中，通常会对数据作均值，提高速度和精度。在caffe中同样会有这样的操作，所以在这里对数据作均值处理。在caffe中提供了一个计算均值的文件compute_image_mean.cpp，在这里直接使用即可。

```
# sudo build/tools/compute_image_mean examples/myfile/img_train_lmdb examples/myfile/mean.binaryproto
```

- compute_image_mean参数说明：总共有两个参数，第一个参数是lmdb训练数据位置，第二个参数设定均值文件的名字及保存路径
- 运行成功后，会在 examples/myfile/ 下面生成一个mean.binaryproto的均值文件。

## 4. 创建模型并编写配置文件

模型就用程序自带的caffenet模型，位置在 models/bvlc_reference_caffenet/文件夹下, 将需要的两个配置文件，复制到myfile文件夹内。

```
# sudo cp models/bvlc_reference_caffenet/solver.prototxt examples/myfile/
# sudo cp models/bvlc_reference_caffenet/train_val.prototxt examples/myfile/
```

复制完成后对配置文件进行修改，即修改solver.prototxt文件。

```
net: "examples/myfile/train_val.prototxt"
test_iter: 2
test_interval: 50
base_lr: 0.001
lr_policy: "step"
gamma: 0.1
stepsize: 100
display: 20
max_iter: 500
momentum: 0.9
weight_decay: 0.005
solver_mode: GPU
```

100个测试数据，batch_size为50，因此test_iter设置为2，就能全cover了。在训练过程中，调整学习率，逐步变小。

修改train_val.protxt，只需要修改两个阶段的data层就可以了，其它可以不用管。

训练层的data层改为：

```
name: "CaffeNet"
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    crop_size: 227
    mean_file: "examples/myfile/mean.binaryproto"
  }
  data_param {
    source: "examples/myfile/img_train_lmdb"
    batch_size: 256
    backend: LMDB
  }
}
```

测试层的data层改为：

```

layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mirror: false
    crop_size: 227
    mean_file: "examples/myfile/mean.binaryproto"
  }
  data_param {
    source: "examples/myfile/img_test_lmdb"
    batch_size: 50
    backend: LMDB
  }
}
```

实际上就是修改两个data layer的mean_file和source这两个地方，其它都没有变化 。

## 5. 训练和测试

通过以上的准备以后，接下来就到最后的阶段了。可以通过以下命令对模型进行训练。

```
# sudo build/tools/caffe train -solver examples/myfile/solver.prototxt
```

结果：

![](https://raw.githubusercontent.com/zhi-z/DeepLearning/master/caffe/image/train_out.png)



## 6.参考文献

[Caffe学习系列(12)：训练和测试自己的图片](https://www.cnblogs.com/denny402/p/5083300.html)