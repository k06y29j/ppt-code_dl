import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

"""
torch.save(vgg16.state_dict(),"vgg16_dict.pth")
model=torch.load("vgg16_dict.pth")
vgg16=torchvision.models.vgg16()
vgg16.load_state_dict(model)
"""
# 加载模型
model=torch.load("vgg16_2.pth")
print(model)

# 检测设备
device = next(model.parameters()).device
print(f"模型所在设备: {device}")
dataset_test=torchvision.datasets.CIFAR10("../data",train=False,
                                      transform=torchvision.transforms.ToTensor(),
                                      download=True)
dataloader_test=DataLoader(dataset_test,batch_size=64)
writer=SummaryWriter("logs/logs_save_load")
step=0
loss=nn.CrossEntropyLoss()
for data in dataloader_test:
    inputs, labels = data
    # 将输入数据移动到与模型相同的设备上
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model.forward(inputs)
    loss_value = loss(outputs, labels)
    writer.add_scalar("loss",loss_value,step)
    step+=1




