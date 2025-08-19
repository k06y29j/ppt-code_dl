import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("../data",train=False,
                                     transform=torchvision.transforms.ToTensor(),
                                     download=True)
dataloader=DataLoader(dataset,batch_size=64)
writer=SummaryWriter("logs/logs_seq")
step=0

class myseq(nn.Module):
    def __init__(self):
        super(myseq,self).__init__()
        self.conv2d1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.conv2d2=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.conv2d3=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2)
        self.maxpool2d=nn.MaxPool2d(kernel_size=2,stride=2)
        self.linear1=nn.Linear(64*4*4,64)
        self.linear2=nn.Linear(64,10)
        self.seq=nn.Sequential(
            self.conv2d1,
            self.maxpool2d,
            self.conv2d2,
            self.maxpool2d,
            self.conv2d3,
            self.maxpool2d,
            nn.Flatten(),
            self.linear1,
            self.linear2
        )
        
    def forward(self,x):
        # x=self.conv2d1(x)
        # x=self.maxpool2d(x)
        # x=self.conv2d2(x)
        # x=self.maxpool2d(x)
        # x=self.conv2d3(x)
        # x=self.maxpool2d(x)
        # x=x.reshape(x.shape[0],-1)
        # x=self.linear1(x)
        # x=self.linear2(x)
        x=self.seq(x)
        return x
    
model=myseq()
# print(model)
# input=torch.rand(64,3,32,32)
# output=model(input)
# print(output.shape)

# writer.add_graph(model,input)
# writer.close()
loss=nn.CrossEntropyLoss()
device=torch.device("cuda:2" if torch.cuda.is_available() and torch.cuda.device_count() > 2 else "cpu")
optimizer=torch.optim.SGD(model.parameters(),lr=0.05)
model=model.to(device)
for epoch in range(20):
    print("epoch:",epoch)
    model.train()
    run_loss=0.0
    for data in dataloader:
        optimizer.zero_grad()
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss_seq=loss(outputs,labels)
        loss_seq.backward()
        optimizer.step()
        run_loss+=loss_seq.item()
    print("loss:",run_loss/len(dataloader))
    writer.add_scalar("run_loss",run_loss/len(dataloader),epoch)
writer.close() 

    