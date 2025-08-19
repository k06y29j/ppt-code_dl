import torch
import torch.nn as nn
import torch.nn.functional as F
#nn.Linear(输入维度，输出维度),nn.ReLU()是激活函数,nn.Sequential()是顺序容器，将各个层按顺序组合起来

net=nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10)) 
X=torch.rand(2,20)
#net(X)是net的forward函数，X是输入，net(X)是输出
print("输入X的形状:", X.shape)
print("输出net(X)的形状:", net(X).shape)
print("输出值:")
print(net(X))

"""自定义块"""
class MLP(nn.Module):
    def __init__(self):
        #调用父类构造函数
        super().__init__()
        self.hidden=nn.Linear(20,256)
        self.out=nn.Linear(256,10)
    def forward(self,X):
        #self.out(F.relu(self.hidden(X)))是输出层，F.relu(self.hidden(X))是隐藏层,F.relu是激活函数
        return self.out(F.relu(self.hidden(X)))

net=MLP()
print("输出net(X)的形状:", net(X).shape)
print("输出值:")
print(net(X))

print("-------------顺序块-------------------")
class MySequential(nn.Module):
    #*args是可变参数，可以传入多个参数
    def __init__(self,*args):
        super().__init__()
        #将参数args中的每个元素添加到self._modules中,enumerate是枚举，将args中的每个元素添加到self._modules中
        for idx,module in enumerate(args):
            self.add_module(str(idx),module)
    #forward函数是顺序块的forward函数，X是输入，self._modules.values()是顺序块中的每个模块，module(X)是每个模块的输出
    def forward(self,X):
        #self._modules.values()是顺序块中的每个模块，module(X)是每个模块的输出
        for module in self._modules.values():
            X=module(X)
        return X

net=MySequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
print("输出net(X)的形状:", net(X).shape)
print("输出值:")
print(net(X))

print("-------------在前向传播函数中执行代码-------------------")
X=torch.rand(2,20)
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        #torch.rand((20,20),requires_grad=False)是随机初始化一个20x20的矩阵，requires_grad=False表示不需要计算梯度
        #rand是随机初始化，randn是正态分布随机初始化
        self.rand_weight=torch.rand((20,20),requires_grad=False)
        #nn.Linear(20,20)是线性层，输入维度为20，输出维度为20
        self.linear=nn.Linear(20,20)
    def forward(self,X):
        #X=self.linear(X)是线性层，X是输入，self.linear(X)是输出
        X=self.linear(X)
        #torch.mm(X,self.rand_weight)是矩阵乘法，X是输入，self.rand_weight是随机初始化的矩阵，torch.mm(X,self.rand_weight)是输出
        X=F.relu(torch.mm(X,self.rand_weight)+1)
        #X=self.linear(X)是线性层，X是输入，self.linear(X)是输出
        X=self.linear(X)
        #X.abs().sum()>1是判断X的绝对值之和是否大于1，如果大于1，则将X除以2
        while X.abs().sum()>1:
            X/=2
        return X.sum()
net=FixedHiddenMLP()
print("输出net(X)的形状:", net(X).shape)
print("输出值:")
print(net(X))
#嵌套块
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

print("-------------嵌套块演示-------------------")
X = torch.rand(2, 20)
print("初始输入X形状:", X.shape)

# 创建嵌套块
nest_mlp = NestMLP()
print("NestMLP内部结构:")
print("  - self.net: Sequential(Linear(20,64), ReLU, Linear(64,32), ReLU)")
print("  - self.linear: Linear(32,16)")

# 演示嵌套过程
print("\n数据流过程:")
print("1. 输入X形状:", X.shape)  # [2, 20]

# 通过self.net (Sequential块)
temp1 = nest_mlp.net(X)
print("2. 经过self.net后形状:", temp1.shape)  # [2, 32]

# 通过self.linear
output = nest_mlp.linear(temp1)
print("3. 经过self.linear后形状:", output.shape)  # [2, 16]

print("\n最终输出:")
print(output)

print("\n-------------更复杂的嵌套: chimera-------------------")
chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
print("chimera结构:")
print("  - 第1层: NestMLP() (20->16)")
print("  - 第2层: Linear(16, 20)")
print("  - 第3层: FixedHiddenMLP() (输出标量)")

result = chimera(X)
print("chimera输出形状:", result.shape)
print("chimera输出值:", result)

print("\n-------------Python列表存储模块的问题-------------------")

class MySequentialWithList(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # 使用Python列表存储模块
        self.modules_list = list(args)
    
    def forward(self, X):
        for module in self.modules_list:
            X = module(X)
        return X

# 测试使用Python列表的版本
print("使用Python列表存储模块:")
net_list = MySequentialWithList(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print("网络结构:", net_list)
print("输出形状:", net_list(X).shape)

print("\n问题演示:")
print("1. 参数不会自动注册到模型中")
print("2. 无法通过net.parameters()获取参数")
print("3. 无法通过net.state_dict()保存模型")

# 对比正确的版本
print("\n正确的MySequential版本:")
net_correct = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print("网络结构:", net_correct)
print("输出形状:", net_correct(X).shape)

print("\n参数对比:")
print("Python列表版本参数数量:", len(list(net_list.parameters())))
print("正确版本参数数量:", len(list(net_correct.parameters())))

print("\n状态字典对比:")
print("Python列表版本state_dict键:", list(net_list.state_dict().keys()))
print("正确版本state_dict键:", list(net_correct.state_dict().keys()))

print("\n-------------参数访问和打印详解-------------------")

# 创建一个网络用于演示
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
X = torch.rand(2, 20)

print("完整网络结构:")
print(net)

print("\n1. net[0] 是什么:")
print("net[0] =", net[0])
print("类型:", type(net[0]))

print("\n2. net[0].named_parameters() 返回什么:")
named_params = net[0].named_parameters()
print("named_params类型:", type(named_params))
print("named_params内容:")
for name, param in named_params:
    print(f"  {name}: {param.shape}")

print("\n3. 列表推导式 [(name,param.shape) for name,param in net[0].named_parameters()]:")
param_list = [(name, param.shape) for name, param in net[0].named_parameters()]
print("param_list:", param_list)

print("\n4. 解包操作 *param_list:")
print("解包前:", param_list)
print("解包后相当于:", *param_list)

print("\n5. 最终结果:")
print(*[(name, param.shape) for name, param in net[0].named_parameters()])

print("\n6. 对比不同写法:")
print("写法1 (解包):", *[(name, param.shape) for name, param in net[0].named_parameters()])
print("写法2 (列表):", [(name, param.shape) for name, param in net[0].named_parameters()])
print("写法3 (循环):")
for name, param in net[0].named_parameters():
    print(f"  {name}: {param.shape}")

print("\n7. 访问不同层的参数:")
print("net[0] (第1层) 参数:")
print(*[(name, param.shape) for name, param in net[0].named_parameters()])

print("net[2] (第3层) 参数:")
print(*[(name, param.shape) for name, param in net[2].named_parameters()])

print("整个网络参数:")
print(*[(name, param.shape) for name, param in net.named_parameters()])

