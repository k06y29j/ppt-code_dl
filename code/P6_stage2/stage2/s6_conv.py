from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10("../data",train=False,
                                     transform=torchvision.transforms.ToTensor(),
                                     download=True)
dataloader=DataLoader(dataset,batch_size=64)

class myconv():
    def __init__(self):
        super(myconv,self).__init__()
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x=self.conv1(x)
        return x

model=myconv()

writer=SummaryWriter("logs")
step=0
for data in dataloader:
    inputs, labels = data
    outputs = model.forward(inputs)
    writer.add_images("input",inputs,step)
    writer.add_images("output",outputs.reshape(-1,3,30,30),step)
    step+=1