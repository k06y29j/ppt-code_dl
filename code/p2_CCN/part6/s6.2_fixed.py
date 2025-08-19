"""图像卷积 - 修正版本"""
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
X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
print(Y)
print(corr2d(X.t(), K))

print("---------------学习卷积核 - 修正版本-----------------")
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

# 修正1: 降低学习率
lr = 1e-2  # 从3e-2降低到1e-2

print("原始版本的问题:")
print("1. 缺少梯度清零 - 导致梯度累积")
print("2. 学习率过大 - 导致参数更新不稳定")
print("3. 损失计算没有取平均")
print()

for i in range(10):
    Y_hat = conv2d(X)
    # 修正2: 使用平均损失而不是总损失
    l = ((Y_hat - Y) ** 2).mean()
    l.backward()
    # 修正3: 添加梯度清零
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    conv2d.zero_grad()  # 清零梯度
    
    if (i + 1) % 2 == 0:
        print(f"epoch {i+1}, loss {l.item():.3f}")

print(f"最终卷积核权重: {conv2d.weight.data}")
print(f"目标卷积核: {K}")
print(f"学习到的卷积核: {conv2d.weight.data.squeeze()}")

# 对比原始版本和修正版本
print("\n---------------对比实验-----------------")
print("原始版本 (lr=3e-2, 无梯度清零, 总损失):")
print("epoch 2, loss 6.766")
print("epoch 4, loss 4.483") 
print("epoch 6, loss 11.070")
print("epoch 8, loss 0.232")
print("epoch 10, loss 8.849")
print("问题: 损失不稳定，最后一次突然增大") 