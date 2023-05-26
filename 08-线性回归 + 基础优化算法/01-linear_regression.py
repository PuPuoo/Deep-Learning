'''
Author: PuPuoo
Date: 2023-05-26 15:18:25
LastEditors: PuPuoo
LastEditTime: 2023-05-26 17:55:46
FilePath: \deep-learning\08-线性回归 + 基础优化算法\01-linear_regression.py
Description: 线性回归的从零开始实现
'''

import matplotlib.pyplot as plt
import random
import torch
from d2l import torch as d2l

# 1.生成数据集

'''
description: 根据带有噪声的线性模型构造一个人造数据集
param {*} w
param {*} b
param {*} num_examples
return {*}
'''
def synthetic_data(w, b, num_examples):
    """ 生成y = Xw + b + 噪声 """
    X = torch.normal(0,1,(num_examples,len(w)))
    y = torch.matmul(X,w) + b
    y += torch.normal(0,0.01,y.shape)
    return X, y.reshape((-1,1))

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = synthetic_data(true_w,true_b,1000) # features中的每⼀⾏都包含⼀个⼆维数据样本，labels中的每⼀⾏都包含⼀维标签值（⼀个标量）
print('features:',features[0],'\nlabel:',labels[0])

# 通过⽣成第⼆个特征features[:, 1]和labels的散点图，可以直观观察到两者之间的线性关系
d2l.set_figsize()
d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)
# plt.show() # 绘图


# 2.读取数据集

# 训练模型时要对数据集进⾏遍历，每次抽取⼀⼩批量样本，并使⽤它们来更新我们的模型
'''
description: ⽣成⼤⼩为batch_size的⼩批量,每个⼩批量包含⼀组特征和标签
param {*} batch_size 接收批量大小
param {*} features 特征矩阵
param {*} labels 标签向量
return {*}
'''
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples)) # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]

# 读取第⼀个⼩批量数据样本并打印。每个批量的特征维度显⽰批量⼤⼩和输⼊特征数。同样的，批量的标签形状与batch_size相等
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n',y)
    break


# 3.初始化模型参数

w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True) # 每次更新都需要计算损失函数关于模型参数的梯度。有了这个梯度，我们就可以向减⼩损失的⽅向更新每个参数


# 4.定义模型

'''
description: 定义模型
param {*} X
param {*} w
param {*} b
return {*}
'''
def linreg(X, w, b):
    """ 线性回归模型 """
    return torch.matmul(X, w) + b # 注意，上⾯的Xw是⼀个向量,当我们⽤⼀个向量加⼀个标量时,标量会被加到向量的每个分量上


# 5.定义损失函数

'''
description: 定义损失函数
param {*} y_hat
param {*} y
return {*}
'''
def squared_loss(y_hat, y):
    """ 均方损失 """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 6.定义优化算法

'''
description: 定义优化算法
param {*} params 参数集合
param {*} lr 学习速率
param {*} batch_size 批量大小
return {*}
'''
def sgd(params, lr, batch_size):
    """ 小批量随机梯度下降 """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 7.训练

# 迭代周期个数num_epochs和学习率lr都是超参数
lr = 0.03
num_epochs = 3
net = linreg # 线性回归模型
loss = squared_loss # 计算损失函数

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y) # X和y的小批量损失
        # 因为l形状是(batch_size,1)，⽽不是⼀个标量。l中的所有元素被加到⼀起
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size) # 使用参数的梯度更细你参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):.5f}') # 以f开头，包含的{}表达式在程序运行时会被表达式的值代替 “:f”的作用是以标准格式输出一个浮点数

print(f'w的估计误差:{true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差:{true_b - b}')





