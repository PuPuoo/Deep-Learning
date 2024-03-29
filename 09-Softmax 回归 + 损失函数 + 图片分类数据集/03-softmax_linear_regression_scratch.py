'''
Author: PuPuoo
Date: 2023-05-29 13:28:22
LastEditors: PuPuoo
LastEditTime: 2023-05-29 18:53:08
FilePath: \deep-learning\09-Softmax 回归 + 损失函数 + 图片分类数据集\03-softmax_linear_regression_scratch.py
Description: softmax回归的简洁实现
'''

import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 1.初始化模型参数

# PyTorch不会隐式地调整输⼊的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整⽹络输⼊的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01) # 以均值0和标准差0.01随机初始化权重

net.apply(init_weights)


# 2.重新审视softmax的实现

loss = nn.CrossEntropyLoss(reduction='none')


# 3.优化算法

trainer = torch.optim.SGD(net.parameters(), lr=0.1) # 使⽤学习率为0.1的⼩批量随机梯度下降作为优化算法


# 4.训练

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)










