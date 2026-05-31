import torch
import numpy as np
import random

#1.创建线性空间的张量
def test01():
    #1.在指定区间按照步长生成元素 [start,end,step)
    data = torch.arange(0,10,2)  #0,2,4,6,8
    print(data)
    
    #2.在指定区间按照元素个数生成
    data = torch.linspace(0,11,10)     #开头是0，结尾是11，元素个数是10
    print(data)

#2.创建随机张量
def test02():
    #1.创建随机张量
    data = torch.randn(2,3)     #创建2行3列张量
    print(data)

    #2.随机数种子设置
    print('随机数种子：',torch.random.initial_seed())
    torch.random.manual_seed(100)     
    print('随机数种子：',torch.random.initial_seed())

#3.随机种子的复现作用测试
def test03():
    #1.设置随机种子
    torch.random.manual_seed(1)

    #2.生成随机张量
    tensor1 = torch.randn(2,3)
    print("Tensor 1:\n",tensor1) 

    #3.再次设置相同的随机种子   
    torch.random.manual_seed(1)

    #4.再次创建一个随机初始化的张量
    tensor2 = torch.randn(2,3)
    print("Tensor 2:\n",tensor2)

    #5.检查两个张量是否相同
    print("Are tensors equal?", torch.equal(tensor1, tensor2))

def test04():
    data1 = torch.tensor(np.random.randn(2,3))
    print(data1)

    data2 = torch.randn(2,3)
    print(data2)

    
if __name__ == "__main__":
     test01()
    # test02()    
    # test03()
    # test04()
