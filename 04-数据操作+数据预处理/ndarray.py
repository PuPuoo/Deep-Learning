'''
Author: PuPuoo
Date: 2023-05-23 14:03:48
LastEditors: PuPuoo
LastEditTime: 2023-05-23 21:23:10
FilePath: \deep-learning\04-数据操作+数据预处理\ndarray.py
Description: 
'''
import torch

# 1.张量的创建

t = torch.ones(4) # ones 函数创建一个具有指定形状的新张量，并将所有元素值设置为1
print('t:',t)

x = torch.arange(12)
print('x:',x)
print('x shape:',x.shape) # 访问向量的形状

y = x.reshape(3,4) # 改变一个张量的形状而不改变元素数量和元素值
print('y:',y)
print('y.numel():',y.numel()) # 返回张量中元素的总个数

z = torch.zeros(2,3,4) # 创建一个张量，其中所有元素都设置为0  两层3x4矩阵
print('z:',z)

w = torch.randn(2,3,4) # 每个元素都从均值为0、标准差为1的标准高斯（正态）分布中随机采样
print('w:',w)

q = torch.tensor([[1,2,3],[4,5,6],[7,8,9]]) # 通过提供包含数值的python列表来为所需张量中的每个元素赋予确定值
print('q:',q)


# 2.张量的运算

x = torch.tensor([1.0,2,4,8])
y = torch.tensor([2,2,2,2])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y)
print(torch.exp(x))

X = torch.arange(12,dtype=torch.float32).reshape(3,4)
Y = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
print('cat操作 dim=0',torch.cat((X,Y),dim=0)) # 按行叠放，已形成更大的张量
print('cat操作 dim=1',torch.cat((X,Y),dim=1)) # 按列叠放，已形成更大的张量

print('X == Y',X == Y) # 通过逻辑运算符构建二元张量
print('X < Y',X < Y)
print('张量所有元素的和：',X.sum()) # 张量所有元素的和


# 3.广播机制

a = torch.arange(3).reshape(3,1)
b = torch.arange(2).reshape(1,2)
print('a:',a)
print('b:',b)
print('a + b:',a + b) # 神奇的广播运算


# 4.索引和切片

X = torch.arange(12,dtype=torch.float32).reshape(3,4)
print('X:',X)
print('X[-1]:',X[-1]) # 最后一个元素，这里指的是一行
print('X[1:3]:',X[1:3]) # 选择第二个和第三个元素，这里指第二行和第三行

X[1,2] = 9 # 第2行第三列改变元素为9
print('X:',X)

X[0:2,:] = 12 # 从第0行到第1行，从第0列到最后改变元素为12
print('X:',X)


# 5.节约内存
before = id(Y) # id函数提供了内存中引用对象的确切地址
Y = Y + X
print(id(Y) == before)

before = id(X)
X += Y
print(id(X) == before)

before = id(X)
X[:] = X + Y
print(id(X) == before) # 使用X[:] = X + Y 或 X += Y来减少操作的内存开销


# 6.转换为其他python对象

Y = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
A = Y.numpy()
print(type(A)) # 打印A的类型
print(A)
B = torch.tensor(A)
print(type(B)) # 打印B的类型
print(B)

a = torch.tensor([3.5])
print(a,a.item(),float(a),int(a))




