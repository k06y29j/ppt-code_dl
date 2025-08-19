import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter

input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]],dtype=torch.float32).reshape(1,1,5,5)

dataset=torchvision.datasets.CIFAR10("../data",train=False,
                                     transform=torchvision.transforms.ToTensor(),
                                     download=True)
dataloader=DataLoader(dataset,batch_size=64)
writer=SummaryWriter("logs/logs_pool")
step=0

class mypool(nn.Module):
    def __init__(self):
        super(mypool,self).__init__()
        self.pool=nn.MaxPool2d(kernel_size=3,padding=0,ceil_mode=False)

    def forward(self,x):
        x=self.pool(x)
        return x
    
pool=mypool()
print(pool(input))

# for data in dataloader:
#     inputs, labels = data
#     outputs = pool.forward(inputs)
#     writer.add_images("input",inputs,step)
#     writer.add_images("output",outputs.reshape(-1,3,30,30),step)
#     step+=1