"""自定义层"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        #mean() 计算张量中所有元素的平均值
        return x-x.mean()

layer=CenteredLayer()
print(layer(torch.FloatTensor([1,2,3,4,5])))
net=nn.Sequential(nn.Linear(8,128),CenteredLayer())
Y=net(torch.rand(4,8))
print(Y.mean())

print("---------带参数的层-----------------")
class MyLinear(nn.Module):
    def __init__(self,in_units,units):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(in_units,units))
        self.bias=nn.Parameter(torch.randn(units,))
    def forward(self,X):
        linear=torch.matmul(X,self.weight.data)+self.bias.data
        return F.relu(linear)
    
linear=MyLinear(5,3)
print(linear.weight)
print(linear(torch.rand(2,5)))

net=nn.Sequential(MyLinear(64,8),MyLinear(8,1))
print(net(torch.rand(2,64)))