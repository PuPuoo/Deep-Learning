'''
Author: PuPuoo
Date: 2023-05-26 19:21:33
LastEditors: PuPuoo
LastEditTime: 2023-05-26 22:03:15
FilePath: \deep-learning\09-Softmax 回归 + 损失函数 + 图片分类数据集\01-images.py
Description: 图片分类数据集
'''

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

# 1.读取数据集

# 可以通过框架中的内置函数将Fashion-MNIST数据集下载并读取到内存中
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式
# 并除以255使得所有像素的数值均在0〜1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)

""" Fashion-MNIST由10个类别的图像组成，每个类别由训练数据集（train dataset）中的6000张图像和测试数据
集（test dataset）中的1000张图像组成。因此，训练集和测试集分别包含60000和10000张图像。测试数据集
不会⽤于训练，只⽤于评估模型性能。 """
print(len(mnist_train))
print(len(mnist_test))

""" 每个输⼊图像的⾼度和宽度均为28像素。数据集由灰度图像组成，其通道数为1。为了简洁起⻅，本书将⾼
度h像素、宽度w像素图像的形状记为h x w或（h,w） """
print(mnist_train[0][0].shape)

'''
description: Fashion-MNIST中包含的10个类别，分别为t-shirt（T恤）、trouser（裤⼦）、pullover（套衫）、dress（连⾐
            裙）、coat（外套）、sandal（凉鞋）、shirt（衬衫）、sneaker（运动鞋）、bag（包）和ankle boot（短靴）。
            以下函数⽤于在数字标签索引及其⽂本名称之间进⾏转换。
param {*} labels
return {*}
'''
def get_fashion_mnist_labels(labels):
    """ 返回Fashion-MNIST数据集的文本标签 """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

'''
description: 创建⼀个函数来可视化这些样本
param {*} imgs
param {*} num_rows
param {*} num_cols
param {*} titles
param {*} scale
return {*}
'''
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """ 绘制图像列表 """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.show() # 绘图
    return axes

# 训练数据集中前⼏个样本的图像及其相应的标签
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))


# 2.读取小批量

batch_size = 256

def get_dataloader_workers():
    """ 使用4个进程来读取数据 """
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())
# 读取训练数据所需的时间
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')


# 3.整合所有组件

'''
description: ⽤于获取和读取Fashion-MNIST数据集
param {*} batch_size 
param {*} resize 将图像⼤⼩调整为另⼀种形状
return {*} 训练集和验证集的数据迭代器
'''
def load_data_fashion_mnist(batch_size, resize=None):
    """ 下载Fashion-MNIST数据集，然后将其加载到内存中 """
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()), 
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))

train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break