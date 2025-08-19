"""图像卷积"""
import torch
from torch import nn
from d2l import torch as d2l
# 定义互相关运算
def corr2d(X, K):
    """计算二维互相关运算"""
    h, w = K.shape
    # 输出Y大小：(X.shape[0] - h + 1, X.shape[1] - w + 1)
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # 计算卷积：X[i:i + h, j:j + w] * K
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))
print("---------------卷积层-----------------")
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        # 随机初始化卷积核, 卷积核大小为kernel_size,Parameter表示这是一个可训练的参数
        self.weight = nn.Parameter(torch.randn(kernel_size))
        # 随机初始化偏置
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

print("---------------图像中目标的边缘检测-----------------")
X=torch.ones((6,8))
X[:,2:6]=0
print(X)
K=torch.tensor([[1.0,-1.0]])
Y=corr2d(X,K)
print(Y)
print(corr2d(X.t(),K))

print("---------------学习卷积核-----------------")
conv2d=nn.Conv2d(1,1,kernel_size=(1,2),bias=False)
X=X.reshape((1,1,6,8))
Y=Y.reshape((1,1,6,7))
lr=3e-2
for i in range(10):
    Y_hat=conv2d(X)
    l=((Y_hat-Y)**2)
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:]-=lr*conv2d.weight.grad
    if(i+1)%2==0:
        print(f"epoch {i+1}, loss {l.sum():.3f}")
print(conv2d.weight.data)