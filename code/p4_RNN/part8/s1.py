"""序列模型"""

import torch
from torch import nn
from d2l import torch as d2l

T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
# 生成正弦信号,torch.normal(0, 0.2, (T,)) 生成均值为0，方差为0.2的正态分布的随机数
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
d2l.plt.savefig('part8/s1_img/sin_time.jpg')
d2l.plt.close()