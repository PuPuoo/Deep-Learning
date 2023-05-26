'''
Author: PuPuoo
Date: 2023-05-26 18:05:07
LastEditors: PuPuoo
LastEditTime: 2023-05-26 18:45:05
FilePath: \deep-learning\08-线性回归 + 基础优化算法\02-linear_regression_scratch.py
Description: 线性回归的简洁实现
'''

import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

# 1.生成数据集

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# 2.读取数据集

'''
description: 我们可以调⽤框架中现有的API来读取数据。我们将features和labels作为API的参数传递,并通过数据迭代
             器指定batch_size。此外,布尔值is_train表⽰是否希望数据迭代器对象在每个迭代周期内打乱数据。
param {*} data_arrays
param {*} batch_size
param {*} is_train
return {*}
'''
def load_array(data_arrays, batch_size, is_train=True):
    """ 构造一个PyTorch数据迭代器 """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
# 这⾥我们使⽤iter构造Python迭代器，并使⽤next从迭代器中获取第⼀项
print(next(iter(data_iter)))


# 3.定义模型

""" 
    我们⾸先定义⼀个模型变量net,它是⼀个Sequential类的实例。Sequential类将多个层串联在⼀起。
    当给定输⼊数据时,Sequential实例将数据传⼊到第⼀层,然后将第⼀层的输出作为第⼆层的输⼊，以此类推。
"""
# nn是神经⽹络的缩写
from torch import nn

net = nn.Sequential(nn.Linear(2, 1)) # 我们将两个参数传递到nn.Linear中。第⼀个指定输⼊特征形状，即2，第⼆个指定输出特征形状，输出特征形状为单个标量，因此为1


# 4.初始化模型参数

net[0].weight.data.normal_(0, 0.01) # net[0]选择⽹络中的第⼀个图层,然后使⽤weight.data和bias.data⽅法访问参数,使⽤替换⽅法normal_和fill_来重写参数值
net[0].bias.data.fill_(0)


# 5.定义损失函数

loss = nn.MSELoss() # 计算均⽅误差使⽤的是MSELoss类，也称为平⽅L2范数。默认情况下，它返回所有样本损失的平均值


# 6.定义优化算法

trainer = torch.optim.SGD(net.parameters(), lr=0.03) # PyTorch在optim模块中实现了该算法的许多变种
# 当我们实例化⼀个SGD实例时，我们要指定优化的参数（可通过net.parameters()从我们的模型中获得）以及优化算法所需的超参数字典。
# ⼩批量随机梯度下降只需要设置lr值，这⾥设置为0.03


# 7.训练

"""
    对于每一个小批量，我们会进行以下步骤：
    1. 通过调⽤net(X)⽣成预测并计算损失l（前向传播）
    2. 通过进⾏反向传播来计算梯度
    3. 通过调⽤优化器来更新模型参数
"""
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward() 
        trainer.step() # 更新参数 w和b，然后进行下一批次的训练
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

