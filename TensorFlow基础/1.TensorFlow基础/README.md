# TensorFlow 基础

- TensorFlow基础结构
- 创建常量、变量
- TensorFlow与numpy
- 容器的创建等

## 1 TensorFlow一些概念

- 使用张量（tensor）表示数据；
- 使用图（grapg）来表示计算任务；
- 创建对话（Session）来执行图；
- 通过变量（Variable）维护状态；
- 使用feed和fetch可以任意的操作(arbitrary operation)赋值或者从中获取数据。

## 2 TensorFlow的计算图

### 2.1 TensorFlow两个部分

- 构造部分，包含计算流图
- 执行部分，通过session来执行图中的计算

### 2.2 构建图

- 创建源节点
- 源节点输出传递给其他节点（op）做运算


```python
import tensorflow as tf
```

## 3 创建变量
每次创建变量在使用之前都要对变量进行初始化。

TensorFlow需要显示输出tensor，需要借助eval()函数。


```python
a = 3
# 创建一个变量
w = tf.Variable([[0.5,1.0]])
x = tf.Variable([[2.0],[1.0]]) 

y = tf.matmul(w, x)  

#全局变量初始化
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print (y.eval())
```


## 4 TensorFlow vs numpy

tensorflow很多操作跟numpy有些类似的

* tf.zeros([3, 4], int32)

* tf.zeros_like(tensor)

* tf.ones([2, 3], int32) 

* tf.ones_like(tensor)

* tensor = tf.constant([1, 2, 3, 4, 5, 6, 7])

* tensor = tf.constant(-1.0, shape=[2, 3])                                           

* tf.linspace(10.0, 12.0, 3, name="linspace")

* tf.range(start, limit, delta)


```python
#生成的值服从具有指定平均值和标准偏差的正态分布
norm = tf.random_normal([2, 3], mean=-1, stddev=4)

# 洗牌
c = tf.constant([[1, 2], [3, 4], [5, 6]])
shuff = tf.random_shuffle(c)

# 每一次执行结果都会不同
# 创建对话
sess = tf.Session()

print (sess.run(norm))
print (sess.run(shuff))
```


## 5 加法与赋值操作


```python
state = tf.Variable(0)
# 加法操作，state + 常量1
new_value = tf.add(state, tf.constant(1))
# 赋值操作，把new_value 赋值给state
update = tf.assign(state, new_value)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(state))    
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
```


## 6 numpy转换成TensorFlow变量


```python
import numpy as np
a = np.zeros((3,3))
ta = tf.convert_to_tensor(a)
with tf.Session() as sess:
     print(sess.run(ta))
```


## 7 TensorFlow常量运算


```python
a = tf.constant(5.0)
b = tf.constant(10.0)

x = tf.add(a, b, name="add")
y = tf.div(a, b, name="divide")

with tf.Session() as sess:
    print("a =", sess.run(a))
    print("b =", sess.run(b))
    print("a + b =", sess.run(x))
    print("a/b =", sess.run(y))
```


## 8 容器的使用
先创建一个容器，并指定容器要放的东西，最后往容器中添加所需要的东西


```python
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
# 先定义操作，里面还没有实际的值
output = tf.multiply(input1, input2)
with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
```

