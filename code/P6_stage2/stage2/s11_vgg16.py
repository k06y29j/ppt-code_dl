import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# # 定义数据预处理
# transform_train = transforms.Compose([
#     transforms.RandomResizedCrop(224),  # 随机裁剪到224x224
#     transforms.RandomHorizontalFlip(),  # 随机水平翻转
#     transforms.ToTensor(),  # 转换为Tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet标准化参数
#                          std=[0.229, 0.224, 0.225])
# ])

# transform_val = transforms.Compose([
#     transforms.Resize(256),  # 缩放到256
#     transforms.CenterCrop(224),  # 中心裁剪到224x224
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # 加载训练集
# train_data = torchvision.datasets.ImageFolder(
#     root="/data/small-datasets-1/imagenet/train",
#     transform=transform_train
# )

# # 加载验证集
# val_data = torchvision.datasets.ImageFolder(
#     root="/data/small-datasets-1/imagenet/val",
#     transform=transform_val
# )

# # 创建数据加载器
# batch_size = 64
# train_loader = DataLoader(
#     train_data,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=4,
#     pin_memory=True
# )

# val_loader = DataLoader(
#     val_data,
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=4,
#     pin_memory=True
# )

from torchvision.models import VGG16_Weights

# 使用新的权重参数格式加载预训练模型
vgg16 = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
# 或者使用 DEFAULT 获取最新的权重
# vgg16 = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
print(vgg16)

dataset=torchvision.datasets.CIFAR10("../data",train=True,
                                      transform=torchvision.transforms.ToTensor(),
                                      download=True)
dataloader=DataLoader(dataset,batch_size=64)
writer=SummaryWriter("logs/logs_vgg16")
step=0
loss=nn.CrossEntropyLoss()
device=torch.device("cuda:2" if torch.cuda.is_available() and torch.cuda.device_count() > 2 else "cpu")
optimizer=torch.optim.SGD(vgg16.parameters(),lr=0.01)
vgg16.classifier.add_module("add_linear",nn.Linear(1000,10))
vgg16=vgg16.to(device)
for epoch in range(20):
    print("epoch:",epoch)
    vgg16.train()
    run_loss=0.0
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = vgg16(inputs)
        loss_value = loss(outputs, labels)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        run_loss+=loss_value.item()
    print("loss:",run_loss/len(dataloader))
    writer.add_scalar("loss",run_loss,epoch)
    writer.add_graph(vgg16,input_to_model=inputs)
    
writer.close()

torch.save(vgg16,"vgg16_2.pth")