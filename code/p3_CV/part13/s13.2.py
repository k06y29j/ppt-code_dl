"""微调"""
import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')
#ImageFolder是torchvision.datasets中的一个类，用于加载图像数据集
#os.path.join(data_dir, 'train')是训练集的目录
#os.path.join(data_dir, 'test')是测试集的目录
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))
#train_imgs[i][0]是训练集中的第i个图像
#train_imgs[-i - 1][0]是训练集中的倒数第i个图像，[-i - 1]表示索引从-1开始，-1表示最后一个元素，-2表示倒数第二个元素，以此类推
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
plt.savefig('part13/s13.2_img/hotdog_show.jpg')
plt.close()

# 使用RGB通道的均值和标准差，以标准化每个通道,mean,std
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#随机裁剪、随机水平翻转、转换为张量、标准化
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])
#将图像调整为256x256，然后裁剪为224x224，转换为张量、标准化
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])
print("""定义并初始化模型""")
pretrained_net = torchvision.models.resnet18(pretrained=True)
#fc是全连接层，in_features是输入特征的维度，out_features是输出特征的维度
print(pretrained_net.fc)

finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)

print("""微调模型""")
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        #SGD是随机梯度下降，params_1x是第一组参数，net.fc.parameters()是第二组参数，lr是学习率，weight_decay是权重衰减
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
    
train_fine_tuning(finetune_net, 5e-5)
plt.savefig('part13/s13.2_img/finetune_net_true.jpg')
plt.close()
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)
plt.savefig('part13/s13.2_img/scratch_net_false.jpg')