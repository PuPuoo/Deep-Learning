'''
Author: PuPuoo
Date: 2023-05-24 14:02:15
LastEditors: PuPuoo
LastEditTime: 2023-05-24 16:25:22
FilePath: \deep-learning\04-数据操作+数据预处理\02-pandas.py
Description: 
'''

import os
import pandas as pd 
import torch

os.makedirs(os.path.join('..','data'),exist_ok=True) # 在上级目录创建data文件夹
data_file = os.path.join('..','data','house_tinyy.csv') # 创建文件
with open(data_file,'w') as f: # 往文件中写数据
    f.write('NumRooms,Alley,Price\n') # 列名
    f.write('NA,Pave,127500\n') # 第一行的值
    f.write('2,NA,106000\n') # 第二行的值
    f.write('4,NA,178100\n') # 第三行的值
    f.write('NA,NA,140000\n') # 第四行的值

data = pd.read_csv(data_file) # 可以看到原始表格中的空值NA被识别成了NAN
print('1.原始数据：\n',data)

inputs, outputs = data.iloc[:,0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean()) # 用均值填充NAN
print(inputs)
print(outputs)

# 利用pandas中的get_dummies函数来处理离散值或者类别值
# 对于inputs中的类别值或离散值，我们将“NaN”视为⼀个类别。由于“巷⼦类型”（“Alley”）列只接受两
# 种类型的类别值“Pave”和“NaN”，pandas可以⾃动将此列转换为两列“Alley_Pave”和“Alley_nan”。巷
# ⼦类型为“Pave”的⾏会将“Alley_Pave”的值设置为1，“Alley_nan”的值设置为0。缺少巷⼦类型的⾏会
# 将“Alley_Pave”和“Alley_nan”分别设置为0和1。
inputs = pd.get_dummies(inputs, dummy_na=True)
print('2.利用pandas中的get_dummies函数处理:\n', inputs)

x,y = torch.tensor(inputs.values),torch.tensor(outputs.values)
print('3.转换为张量:')
print(x)
print(y)

