'''
Author: PuPuoo
Date: 2023-05-27 14:38:06
LastEditors: PuPuoo
LastEditTime: 2023-05-27 16:26:35
FilePath: \deep-learning\09-Softmax 回归 + 损失函数 + 图片分类数据集\02-softmax_linear_regression.py
Description: softmax回归的从零开始
'''

import matplotlib.pyplot as plt
import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 1.初始化模型参数

num_inputs = 784 # 原始数据集中的每个样本都是28×28的图像。本节将展平每个图像，把它们看作⻓度为784的向量
num_outputs = 10 # 因为我们的数据集有10个类别，所以⽹络输出维度为10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)


# 2.定义softmax操作

X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(X.sum(0, keepdim=True), X.sum(1, keepdim=True)) # keepdims=True 用途：保持原数组的维度

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition # 这里应用了广播机制

X = torch.normal(0, 1, (2,5))
X_prob = softmax(X)
print('X_prob:',X_prob ,'X_prob.sum(1):', X_prob.sum(1))


# 3.定义模型

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


# 4.定义损失函数

# 引入交叉熵损失函数
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(y_hat[[0, 1], y])

# 只需⼀⾏代码就可以实现交叉熵损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

print(cross_entropy(y_hat, y))


# 5.分类精度

# 分类精度即正确预测数量与总预测数量之⽐
def accuracy(y_hat, y):
    """ 计算预测正确的数量 """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1) # argmax获得每⾏中最⼤元素的索引来获得预测类别
    cmp = y_hat.type(y.dtype) == y # 将y_hat的数据类型转换为与y的数据类型⼀致,再做"==" 结果是⼀个包含0（错）和1（对）的张量
    return float(cmp.type(y.dtype).sum()) # 求和会得到正确预测的数量

print(accuracy(y_hat, y) / len(y))

# 同样，对于任意数据迭代器data_iter可访问的数据集，我们可以评估在任意模型net的精度
def evaluate_accuracy(net, data_iter):
    """ 计算在指定数据集上模型的精度 """
    if isinstance(net, torch.nn.Module):
        net.eval() # 将模型设置为评估模式
    metric = Accumulator(2) # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

""" Accumulator实例中创建了2个变量，分别⽤于存储正确预测的数量和预测的总数量。当我们遍历数据集
时，两者都将随着时间的推移⽽累加 """
class Accumulator: 
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

print(evaluate_accuracy(net, test_iter))


# 6.训练

def train_epoch_ch3(net, train_iter, loss, updater): # updater是更新模型参数的常⽤函数，它接受批量⼤⼩作为参数。它可以是d2l.sgd函数，也可以是框架的内置优化函数
    """ 训练模型一个迭代周期 """
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用Pytorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


""" 它会在train_iter访问到的训练数据集上训练⼀个模型net。该训练函数将
会运⾏多个迭代周期（由num_epochs指定）。在每个迭代周期结束时，利⽤test_iter访问到的测试数据集对
模型进⾏评估。我们将利⽤Animator类来可视化训练进度 """
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater): 
    """训练模型（定义⻅第3章）"""
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

# ⼩批量随机梯度下降来优化模型的损失函数，设置学习率为0.1
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

# 7.预测

def predict_ch3(net, test_iter, n=6):
    """ 预测标签 """
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)),1, n, titles=titles[0:n])
    plt.show()
    
predict_ch3(net, test_iter)



