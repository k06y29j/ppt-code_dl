"""LeNet"""
import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    #1d卷积层，输入通道1，输出通道6，卷积核大小5，填充2
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))

X=torch.rand(size=(1,1,28,28),dtype=torch.float32)
for layer in net:
    #layer是Sequential中的一个元素，是一个nn.Module对象
    X=layer(X)
    print(layer.__class__.__name__,"output shape:\t",X.shape)

print("--------------训练------------------")
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size=batch_size)
# 评估准确率，net是模型，data_iter是数据迭代器，device是设备
def evaluate_accuracy_gpu(net,data_iter,device=None):
    if isinstance(net,nn.Module):
        net.eval()
        #如果device为None，则使用net的第一个参数的设备
        if not device:
            device=next(iter(net.parameters())).device
    #Accumulator是d2l库中的一个类，用于计算准确率
    metric=d2l.Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            if isinstance(X,list):
                #如果X是list，则将X中的每个元素都转换为device
                X=[x.to(device) for x in X]
            else:
                #如果X不是list，则将X转换为device
                X=X.to(device)
            y=y.to(device)
            #d2l.accuracy是d2l库中的一个函数，用于计算准确率
            metric.add(d2l.accuracy(net(X),y),y.numel())
    #返回准确率
    return metric[0]/metric[1]

def train_ch6(net,train_iter,test_iter,num_epochs,lr,device):
    net.to(device)
    optimizer=torch.optim.SGD(net.parameters(),lr)

#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    #保存图片
    d2l.plt.savefig('part6/lenet.svg')
    d2l.plt.show()
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())