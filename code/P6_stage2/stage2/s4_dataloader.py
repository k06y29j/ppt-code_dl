import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

test_loader= torch.utils.data.DataLoader(
    test_set,
    batch_size=64,
    shuffle=True,
    num_workers=4
)

img,target=test_set[0]
print(img.shape)
print(target)

writer=SummaryWriter('./logs')
for epoch in range(2):
    for i, (images, labels) in enumerate(test_loader):
        writer.add_images('train_images',images,i)
# for i, (images, labels) in enumerate(test_loader):
#     writer.add_images('test_images',images,i)

writer.close() 
    
