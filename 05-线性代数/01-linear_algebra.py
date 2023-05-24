'''
Author: PuPuoo
Date: 2023-05-24 21:22:57
LastEditors: PuPuoo
LastEditTime: 2023-05-24 22:12:28
FilePath: \deep-learning\05-线性代数\01-linear_algebra.py
Description: 
'''

import torch

# 1. 标量与变量

x = torch.tensor([3.0])
y = torch.tensor([2.0])
print(x + y, x * y,x / y,x ** y)


# 2.向量

x = torch.arange(4)
print('x:',x)
print('x[3]:',x[3]) # 通过张量的索引来访问任一元素
print('张量的形状:',x.shape) # 张量的形状
print('张量的长度:',len(x)) # 张量的长度
z = torch.arange(24).reshape(2,3,4)
print('三维张量的长度:',len(z))


# 3.矩阵
A = torch.arange(20).reshape(5,4)
print('A:',A)
print('A.shape:',A.shape)
print('A.shape[-1]:',A.shape[-1])
print('A.T:',A.T) # 矩阵的转置


# 4.矩阵的计算

A = torch.arange(20,dtype=torch.float32).reshape(5,4)
B = A.clone() # 通过分配新内存，将A的一个副本分配给B
print('A:',A)
print('B:',B)
print('A + B:',A + B) # 矩阵相加
print('A * B:',A * B) # 矩阵相乘

a = 2
X = torch.arange(24).reshape(2,3,4)
print('X:',X)
print('a + X:',a + X) # 矩阵的值加上标量
print('a * X:',a * X) # 矩阵的值乘以标量
print((a * X).shape)


# 5.矩阵的sum运算
print('A:',A)
print('A.shape:',A.shape)
print('A.sum():',A.sum())
print('A.sum(axis=0)',A.sum(axis=0)) # 沿0轴汇总已生成输出向量(行汇总)
print('A.sum(axis=1)',A.sum(axis=1)) # 沿1轴汇总已生成输出向量(列汇总)
print('A.sum(axis=1, keepdims=True)',A.sum(axis=1, keepdims=True)) # shape由[5,4]变为[5,1]
print('A.sum(axis=[0,1]):',A.sum(axis=[0,1])) # 与sum一样
print('A.mean():',A.mean()) # 求均值
print('A.sum() / A.numel():',A.sum() / A.numel()) # 求均值


# 6.向量-向量相乘(点积)

x = torch.arange(4,dtype=torch.float32)
y = torch.ones(4,dtype=torch.float32)
print('x:',x)
print('y:',y)
print('向量-向量点积:',torch.dot(x,y))


# 7.矩阵-向量相乘(向量积)

print('A:',A) # 5*4维
print('x:',x) # 4*1维
print('torch.mv(A,x):',torch.mv(A,x))
print('torch.mv(A,x).shape:',torch.mv(A,x).shape)


# 8.矩阵-矩阵相乘(向量积)

print('A:',A) # 5*4维
B = torch.ones(4,3) # 4*3维
print('B:',B)
print('torch.mm(A,B):',torch.mm(A,B))


# 9.范数
u = torch.tensor([3.0,-4.0])
print('向量的L2范数:',torch.norm(u)) # 向量的L2范数
print('向量的L1范数:',torch.abs(u).sum()) # 向量的L1范数
v = torch.ones((4,9))
print('v:',v)
print('矩阵的Frobenius范数:',torch.norm(v))







