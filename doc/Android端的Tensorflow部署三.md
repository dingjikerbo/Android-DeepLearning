---
title: 写给Android程序员的Tensorflow教程三
date: 2018-01-02 19:37:33
---

前面两篇文章都是讨论怎么用别人训好的模型，本篇文章我们将讨论如何训练自己的模型，从最简单的MNIST开始训练，然后创建App识别手写体。

MNIST的训练我们分两种方式，一种初级的采用普通的线性回归，准确率达到91%左右，另一种采用高级点的卷积神经网络，准确率提升到99%。

首先我们看普通的线性回归如何训练模型，参考官方文档，
[MNIST机器学习入门](http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html)

对着python代码来看，这里就是要拟合一个线性函数，y=wx+b，给每张28*28的图展开成1*784的，其实怎么展开不重要，只要所有的图都以相同方式就行。整个代码分为四部分，第一部分读取mnist数据，第二部分定义模型和损失函数，第三部分训练模型，第四部分用测试集测试模型准确度。

我们重点看第四部分，y是模型输出的结果，对于每张图，y都会输出1*10的向量，类似于[0,0,...1,0]这样的形式，只有1位会为1，对应的索引就是识别的结果。tf.argmax(y, 1)的意思是求出每个y向量内最大元素对应的索引，换句话说，就是得到每张图被识别成的数字。然后和标准答案y_进行比较，得到一个bool型的数组，类似于[true, true, false, ...]这样的，每个元素表示每张图被识别对了没有。
然后将bool转成float型，通过tf.reduce_mean将所有元素相加再除以元素个数得到平均准确率。

```
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.int64, [None])

cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Train
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))
```

训练了1000次，准确率也只有91%。

接下来我们看通过卷积神经网络来训模型。可以参考官方文档：
[深入MNIST](http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_pros.html)

代码比较长，我们分成几部分来介绍，首先是读取MNIST数据，

```
#coding=utf-8

from __future__ import print_function
import shutil
import os.path
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

EXPORT_DIR = './model'

if os.path.exists(EXPORT_DIR):
    shutil.rmtree(EXPORT_DIR)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

接下来定义网络，首先是一些基本的参数，

```
# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units
```

然后是基本的网络结构，这里分别是卷积层和池化层。注意到卷积运算后有一个relu。

```
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')
```

接下里是总体的网络结构，

```
# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

# Create Model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
```

输入是很多张[28, 28, 1]的图片，1是因为图片是黑白的。
第一个卷积层输入是[28,28,1]的，经过32个[5,5,1]的卷积核wc1运算输出[28,28,32]，再经过池化层输出为[14,14,32]。

第二个卷积层输入是第一个卷积层的输出，即[14,14,32]，经过64个[5,5,32]的卷积核wc2运算输出[14,14,64]，再经过池化层输出为[7,7,64]。

然后就是一个全连接层了，第二个卷积层的输出[7,7,64]展开成拥有7*7*64=3136个元素的一维数组，然后通过和[3136,1024]矩阵相乘变成只有1024个元素的一维数组。

接下来有个dropout层，指定一个概率。

最后是输出层，类似于一个全连接层，将1024继续降到10，因为输出是10个分类。

接下来是定义损失函数和评估函数，这里pred是预测输出，y是标准输出，二者都是[10]的一维向量。向量中只有1位是1，其余位都为0。因此tf.argmax(y,1)表示y向量中最大的数的索引，即对应着识别出的数。这里判断pred和y识别出的数是否相同，然后转成float，所有图片的结果累积在一起求平均就是最终的准确率。

```
# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
```

接下里开始训练网络，这里每批次取128个数据训，训满20万个数据就结束。然后用test数据中的256张图片测试准确性。注意dropout只在训练时有用，测的时候keep_prob都是1。

最后就是提取各层的参数了，

```
# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                        y: mnist.test.labels[:256],
                                        keep_prob: 1.}))
    WC1 = weights['wc1'].eval(sess)
    BC1 = biases['bc1'].eval(sess)
    WC2 = weights['wc2'].eval(sess)
    BC2 = biases['bc2'].eval(sess)
    WD1 = weights['wd1'].eval(sess)
    BD1 = biases['bd1'].eval(sess)
    W_OUT = weights['out'].eval(sess)
    B_OUT = biases['out'].eval(sess)
```

接下来保存网络，

```
# Create new graph for exporting
g = tf.Graph()
with g.as_default():
    x_2 = tf.placeholder("float", shape=[None, 784], name="input")

    WC1 = tf.constant(WC1, name="WC1")
    BC1 = tf.constant(BC1, name="BC1")
    x_image = tf.reshape(x_2, [-1, 28, 28, 1])
    CONV1 = conv2d(x_image, WC1, BC1)
    MAXPOOL1 = maxpool2d(CONV1, k=2)

    WC2 = tf.constant(WC2, name="WC2")
    BC2 = tf.constant(BC2, name="BC2")
    CONV2 = conv2d(MAXPOOL1, WC2, BC2)
    MAXPOOL2 = maxpool2d(CONV2, k=2)

    WD1 = tf.constant(WD1, name="WD1")
    BD1 = tf.constant(BD1, name="BD1")

    FC1 = tf.reshape(MAXPOOL2, [-1, WD1.get_shape().as_list()[0]])
    FC1 = tf.add(tf.matmul(FC1, WD1), BD1)
    FC1 = tf.nn.relu(FC1)

    W_OUT = tf.constant(W_OUT, name="W_OUT")
    B_OUT = tf.constant(B_OUT, name="B_OUT")

    # skipped dropout for exported graph as there is no need for already calculated weights

    OUTPUT = tf.nn.softmax(tf.matmul(FC1, W_OUT) + B_OUT, name="output")

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    # 返回一个图的序列化的GraphDef表示
    graph_def = g.as_graph_def()
    tf.train.write_graph(graph_def, EXPORT_DIR, 'mnist_model_graph.pb', as_text=False)

    # Test trained model
    y_train = tf.placeholder("float", [None, 10])
    correct_prediction = tf.equal(tf.argmax(OUTPUT, 1), tf.argmax(y_train, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print("check accuracy %g" % accuracy.eval(
            {x_2: mnist.test.images, y_train: mnist.test.labels}, sess))
```


训练完后，输出model文件为model/mnist_model_graph.pb，将其拷贝到assets目录，同时我们自己在assets下创建一个graph_label_strings.txt文件，每一行对应一个输出，分别是0到9。

现在可以测试了，我们创建一个View用于手动绘制的，然后转成Bitmap丢给模型来识别是什么数字。

可参考如下两篇文章：
[Creating Custom Model For Android Using TensorFlow](https://blog.mindorks.com/creating-custom-model-for-android-using-tensorflow-3f963d270bfb)

首先要写Python来训模型，工程可见[MindorksOpenSource/AndroidTensorFlowMNISTExample](https://github.com/MindorksOpenSource/AndroidTensorFlowMNISTExample)

Demo链接 - [Android-DeepLearning/Test3](https://github.com/dingjikerbo/Android-DeepLearning/tree/master/Test3)