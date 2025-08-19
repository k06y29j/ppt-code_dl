"""参数管理"""
import torch
import torch.nn as nn

net=nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
X=torch.rand(2,4)
print(net(X))

print("-------------访问参数，提取参数-------------------")
# 访问参数
print(net[2].state_dict())
print(type(net[2].bias))
# 访问参数
print(net[2].bias)
print(net[2].bias.data)
print(net[2].weight.grad==None)

print("---------------一次性访问所有参数-----------------")
# * 解包,(name,param.shape) 解包后，name和param.shape是两个变量,named_parameters()返回一个生成器
print(*[(name,param.shape) for name,param in net[0].named_parameters()])
# for name, param in net[0].named_parameters():
#     print(f"  {name}: {param.shape}")
#net.named_parameters()返回一个生成器
print(*[(name,param.shape) for name,param in net.named_parameters()])
print(net.state_dict()['2.bias'].data)

print("---------------修从嵌套块收集参数-----------------")
def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(),
    nn.Linear(8,4),nn.ReLU())

def block2():
    net=nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}',block1())
    return net
rgnet=nn.Sequential(block2(),nn.Linear(4,1))
print(rgnet)
print(rgnet[0][1][0].bias.data)

"""参数初始化"""
print("内置初始化")
def init_normal(m):
    if type(m)==nn.Linear:
        # 正态分布初始化
        nn.init.normal_(m.weight,mean=0,std=0.01)
        # 0初始化,初始偏置为0
        nn.init.zeros_(m.bias)
net.apply(init_normal)
print(net[0].weight.data[0],net[0].bias.data[0])

def init_constant(m):
    if type(m)==nn.Linear:
        nn.init.constant_(m.weight,1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
print(net[0].weight.data[0],net[0].bias.data[0])

def init_xavier(m):
    if type(m)==nn.Linear:
        # 均匀分布初始化
        nn.init.xavier_uniform_(m.weight)
net.apply(init_xavier)
print(net[0].weight.data[0])

def init_42(m):
    if type(m)==nn.Linear:
        nn.init.constant_(m.weight,42)
net.apply(init_42)
print(net[0].weight.data[0])

print("---------自定义初始化-----------------")
def my_init(m):
    if type(m)==nn.Linear:
        print("Init",*[(name,param.shape) for name,param in m.named_parameters()][0])
        nn.init.uniform_(m.weight,-10,10)
        m.weight.data*=m.weight.data.abs()>=5
net.apply(my_init)
print(net[0].weight[:2])

print("---------参数绑定-----------------")
shared=nn.Linear(8,8)
net=nn.Sequential(nn.Linear(4,8),nn.ReLU(),
shared,nn.ReLU(),
shared,nn.ReLU(),
nn.Linear(8,1))
print(net)
print(net[2].weight.data[0]==net[4].weight.data[0])