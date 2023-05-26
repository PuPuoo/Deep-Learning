'''
Author: PuPuoo
Date: 2023-05-25 16:33:53
LastEditors: PuPuoo
LastEditTime: 2023-05-25 18:06:28
FilePath: \deep-learning\07-自动求导\01-autograd.py
Description: 
'''

import torch

# 1.一个简单的例子

# x = torch.arange(4.0)
# print('x:',x)
# x.requires_grad_(True) # 需要存储梯度
x = torch.arange(4.0,requires_grad=True) # 以上等价于这个 将梯度附加到想要对其计算偏导数的变量上
print('x:',x)
print('x.grad:',x.grad) # 默认是None

y = 2 * torch.dot(x,x)
print('y:',y) # y为标量
y.backward() # 通过调用反向传播函数来自动计算y关于x每个分量的梯度
print('x.grad:',x.grad)
print(x.grad == 4 * x) # 验证

# 在默认情况下，pytorch会累积梯度，我们需要清楚之前的值
x.grad.zero_()  # 清楚梯度值
print('x.grad:',x.grad)
y = x.sum()
print('y:',y)
y.backward()
print('x.grad:',x.grad)


# 2.非标量变量的反向传播

x.grad.zero_()
y = x * x
print('y:',y)
print('y.sum():',y.sum())
y.sum().backward() # sum():相当于我们在不影响求导结果的前提下，对原来的表达式做了适当的变换，使其计算结果成为了一个标量，再使用这个变换了的表达式对张量里的每一项进行求导。
print('x.grad:',x.grad)


# 3.分离计算

x.grad.zero_()
y = x * x
print('y:',y)
u = y.detach() # 将u作为常数处理
z = u * x 
print('z:',z)
z.sum().backward() # 将u作为常数处理 而不是z = x * x * x关于x的偏导数
print('x.grad:',x.grad)
print(x.grad == u)

x.grad.zero_()
y = x * x * x
y.sum().backward()
print(x.grad == 3*x*x)


# 小结
'''
    深度学习框架可以自动计算导数：我们首先将梯度附加到想要对其计算偏导数的变量上，然后记录目标值的计算，执行它的反向传播函数，并访问得到的梯度
'''






