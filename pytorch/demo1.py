import torch
import numpy as np
import random

#1.根据已有数据创建张量
def test01()

    #1.创建张量biaoliang
    x = 10
    print(x)
    print(type(x))

    data = torch.tensor(10)
    print(data)
    print(type(data))

    #2.numpy数组，由于data为float64,下面代买也使用该类型
    data = np.random.randn(2, 3)
    data = torch.tensor(data)
    print(data)

    #3.列表，下面代码使用默认元素类型float32
    data = [[10., 20., 30.],[40., 50., 60.]]

#2.创建指定形状的张量
def test02():
    #1.创建2行3列的张量，默认dtype为float32
    data = torch.Tensor(2,3)
    print(data)

    #2.注意：如果传递列表，则创建包含指定元素的张量
    data = torch.Tensor([10])
    print(data)

    data = torch.Tensor([10,20])
    print(data)

#3.使用具体类型的张量
def test03():
    #1.创建2行3列，dtype为int32的张量
    data = torch.IntTensor(2,3)
    print(data)

    #2.注意：如果传递的元素类型不正确，则会进行类型转换
    data = torch.IntTensor([2.5,3.3])
    print(data)     

    #3.其他的类型
    data = torch.ShortTensor()    # int16
    data = torch.LongTensor()     # int64
    data = torch.FloatTensor()    # float32
    data = torch.DoubleTensor()   # float64

if __name__ == "__main__":
    # test01()
    # test02()
    test03()
