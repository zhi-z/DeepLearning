# 图像数据转换成db（leveldb/lmdb）文件

## 1 简介

在深度学习的实际项目中，我们经常甬道的原始数据是图片文件，如jpg、png等，而且有可能图片的大小不一致。而在caffe中经常使用的数据类型是lmdb和了leveldb，因此就产生了这样一个问题：如何从原始图片文件转换成caffe中能够运行的db（leveldb/lmdb）文件。

在caffe中，作者为我们提供这样一个文件：convert_imageset.cpp.存放在tools文件夹下，编译之后生成的可执行文集爱你放在buile/tools下面，这个文件的作用就是用于将图片文将图片文件转换成caffe框架中能直接使用的db文件。

## 2 转换格式过程

### 2.1 转换格式 

```
 convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
```

参数说明：

- FLAGS：图片参数组
- ROOTFOLDER/: 图片存放的绝对路径，从linux系统根目录开始
- LISTFILE：图片文件列表清单，一般为一个txt文件，一行一张图片
- DB_NAME:最终的db文件存放目录

### 2.2 文件清单创建（参数LISTFILE）

如果图片与经下载到本地电脑上了，那我们首先需要创建一个图片列表清单，存放在txt文件下。本文以caffe程序中自带的图片为例，进行讲解，图片目录是  example/images/, 两张图片，一张为cat.jpg, 另一张为fish_bike.jpg，表示两个类别。如图所示，在这里我特别复制出多张*cat.jpg文件。

![](/home/datah/Desktop/note/caffe/image/2.png)

接下来通过sh脚本文件，第哦呵用linux命令来生成图片清单：

```
# sudo vi examples/images/create_filelist.sh
```

sh脚本文件中的内容为：

```
# /usr/bin/env sh
DATA=examples/images
echo "Create train.txt..."
rm -rf $DATA/train.txt
find $DATA -name *cat.jpg | cut -d '/' -f3 | sed "s/$/ 1/">>$DATA/train.txt
find $DATA -name *bike.jpg | cut -d '/' -f3 | sed "s/$/ 2/">>$DATA/tmp.txt
cat $DATA/tmp.txt>>$DATA/train.txt
rm -rf $DATA/tmp.txt
echo "Done.."
```

脚本文件中命令说明：

- rm：删除文件
- find：查找文件
- cut：截取路径
- sed：在每行的最后加上标注。本列中中到*cat.jpg文件加入标注1，找到*bike.jpg文件加入标注2
- cat：将两个类别合并在一个文件里

运行：

```
$ bash examples/images/create_filelist.sh
```

结果：生成一个train.txt，文件中的内容为

```
2cat.jpg 1
1cat.jpg 1
cat.jpg 1
fish-bike.jpg 2
```

通过以上的脚本可完成图片文件清单的创建，图片很少的时候，手动编写这个列表清单文件就行了。但图片很多的情况，就需要用脚本文件来自动生成了。在以后的实际应用中，还需要生成相应的val.txt和test.txt文件，方法是一样的。

### 2.3 FLAGS 参数组 

FLAGS参数组的主要参数及作用有：

- -gray: 是否以灰度图的方式打开图片。程序调用opencv库中的imread()函数来打开图片，默认为false
- -shuffle: 是否随机打乱图片顺序。默认为false
- -backend:需要转换成的db文件格式，可选为leveldb或lmdb,默认为lmdb
- -resize_width/resize_height: 改变图片的大小。在运行中，要求所有图片的尺寸一致，因此需要改变图片大小。 程序调用opencv库的resize（）函数来对图片放大缩小，默认为0，不改变
- -check_size: 检查所有的数据是否有相同的尺寸。默认为false,不检查
- -encoded: 是否将原图片编码放入最终的数据中，默认为false
- -encode_type: 与前一个参数对应，将图片编码为哪一个格式：‘png','jpg'......

知道以上的参数以后就可以使用这些命令和参数来生成lmdb数据了。

### 2.4 数据转换成lmdb文件格式

对于参数比较多，可以使用脚本来实现命令。

#### 2.4.1 sh脚本创建

```
# sudo vi examples/images/create_lmdb.sh
```

内容为：

```
#!/usr/bin/en sh
DATA=examples/images
rm -rf $DATA/img_train_lmdb
build/tools/convert_imageset --shuffle \
--resize_height=256 --resize_width=256 \
/home/xxx/caffe/examples/images/ $DATA/train.txt  $DATA/img_train_lmdb
```

参数说明：

设置参数-shuffle,打乱图片顺序。设置参数-resize_height和-resize_width将所有图片尺寸都变为256*256./home/xxx/caffe/examples/images/ 为图片保存的绝对路径。

#### 2.4.2 运行

运行如下命令，就会在examples/images/ 目录下生成一个名为 img_train_lmdb的文件夹，里面的文件就是我们需要的db文件了。

```
# sudo sh examples/images/create_lmdb.sh
```



## 参考文献：

[Caffe学习系列(11)：图像数据转换成db（leveldb/lmdb)文件](https://www.cnblogs.com/denny402/p/5082341.html)

