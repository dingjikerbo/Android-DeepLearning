本文记录如何通过gluon用更少的代码进行线性回归。

首先仍然是创建数据集，和之前一样

```
num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = nd.dot(X, nd.array(true_w)) + true_b
y += 0.01 * nd.random_normal(shape=y.shape)
```

接下来读取数据，之前是自定义一个迭代器，这里的做法如下：

```
batch_size = 10
dataset = gluon.data.ArrayDataset(X, y)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)
```

首先根据X和y生成一个dataset，然后生成对应的迭代器，指定batchSize和是否乱序。用该迭代器读取和之前的一样。

```
for data, label in data_iter:
    print(data, label)
```

接下来定义模型，Sequential用于将所有层串起来，添加一个Dense层，表示线性模型，因为其输入所有结点都与后续的节点相连，这里参数1表示输出节点个数。initialize表示默认随机初始化模型权重。

```
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))
net.initialize()
```

损失函数采用平方误差，

```
square_loss = gluon.loss.L2Loss()
```

接下来定义Trainer，设置模型参数，

```
trainer = gluon.Trainer(
    net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

训练过程如下：

```
epochs = 5
batch_size = 10
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter:
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, average loss: %f" % (e, total_loss/num_examples))
```

此处loss是batch_size行的误差集，nd.sum(loss)返回的是1*1的ndarray，通过asscalar将其转成单个数值，比如将[2.1]转成2.1。

最后打出结果，

```
dense = net[0]
print true_w, dense.weight.data()
print true_b, dense.bias.data()
```

从上可见确实简单多了。
