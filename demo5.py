import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)

        # ⽹络参数初始化（pytorch可以自己实现，可以不用人为定义）
        self.linear1.weight.data = torch.tensor([[0.15, 0.20], [0.25, 0.30]])
        self.linear2.weight.data = torch.tensor([[0.40, 0.45], [0.50, 0.55]])
        self.linear1.bias.data = torch.tensor([0.35, 0.35])
        self.linear2.bias.data = torch.tensor([0.60, 0.60])

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        
        return x

if __name__ == '__main__':
    
    inputs = torch.tensor([[0.05, 0.10]])
    targets = torch.tensor([[0.01, 0.99]])

    #获取网络输出值
    net = Net()
    output = net(inputs)
    #print(output)

    #计算误差
    loss = torch.sum((output - targets) ** 2)/2
    #print(loss)

    #优化方法
    optimizer = optim.SGD(net.parameters(), lr=0.5)

    #梯度清零(必写)
    optimizer.zero_grad()

    #反向传播(必写)
    loss.backward()

    #打印w5、w7、w1的梯度值
    print(net.linear1.weight.grad.data)
    print(net.linear2.weight.grad.data)
    print('-'*50)
    
    #打印网络参数(必写)
    optimizer.step()
    print(net.state_dict())



    