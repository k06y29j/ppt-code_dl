"""序列模型"""

import torch
from torch import nn
from d2l import torch as d2l

T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
print(torch.normal(0, 0.2, (T,)))
# 生成一个正弦波形状的序列，添加一些噪声
# 注意：torch.sin()的输入是弧度制，而不是角度制
# 0.01 * time是将时间转换为弧度制
# torch.normal(0, 0.2, (T,))是添加的噪声
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
d2l.plt.savefig('part8/s1_img/sin_time.jpg')
d2l.plt.close()

# 生成特征和标签
# 这里的tau是时间延迟，表示我们使用前tau个时间步的观测来预测下一个时间步的值
# 特征是一个矩阵，其中每一行包含tau个时间步的观测
# 标签是一个向量，其中每个元素是对应时间步的预测值
tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    # 列i是来自x的观测，其时间步从（i）到（T-tau+i）
    # 例如，当tau=4时，features的第一列是x[0]到x[T-4]，第二列是x[1]到x[T-3]，依此类推
    features[:, i] = x[i: T - tau + i]
# 标签是一个向量，其中每个元素是对应时间步的预测值
# 例如，当tau=4时，标签是x[4]到x[T-1]，即x的第5个到最后一个元素
# 这里的标签是一个列向量，形状为(T-tau, 1)
# 注意：reshape((-1, 1))将标签转换为列向量
# 其中-1表示自动计算行数，1表示列数为1
# 这样做是为了确保标签的形状与特征矩阵的行数一致
# 这样可以方便地将特征和标签配对，用于训练模型
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)

# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 20),
                        nn.ReLU(),
                        nn.Linear(20, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    # 初始化网络权重
    net.apply(init_weights)
    return net

# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')

def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 20, 0.003)

print("------------预测-----------------")
onestep_preds = net(features)
#detach()将张量从计算图中分离出来，避免梯度计算
# numpy()将张量转换为NumPy数组，以便绘图
d2l.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))
d2l.plt.savefig('part8/s1_img/pred_time.jpg')
d2l.plt.close()

print("------------多步预测-----------------")
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))

d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
d2l.plt.savefig('part8/s1_img/multistep_pred_time.jpg')
d2l.plt.close()

print("------------多步预测的误差-----------------")
max_steps = 64

features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))
d2l.plt.savefig('part8/s1_img/multistep_pred_error.jpg')
d2l.plt.close()