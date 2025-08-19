import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter

input=torch.tensor([[1,-0.5],
                    [-1,3]]).reshape(1,1,2,2)

dataset=torchvision.datasets.CIFAR10("../data",train=False,
                                     transform=torchvision.transforms.ToTensor(),
                                     download=True)
dataloader=DataLoader(dataset,batch_size=64)
writer=SummaryWriter("logs/logs_activation")
step=0

class myactivation(nn.Module):
    def __init__(self):
        super(myactivation,self).__init__()
        # self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        # self.tanh=nn.Tanh()
    
    def forward(self,x):
        x=self.sigmoid(x)
        return x
    
activation=myactivation()
print(activation(input))

for data in dataloader:
    inputs, labels = data
    outputs = activation.forward(inputs)
    writer.add_images("input",inputs,step)
    writer.add_images("output",outputs,step)
    step+=1

writer.close()
