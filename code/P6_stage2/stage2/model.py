from torch import nn
import torch

class mynet(nn.Module):
    def __init__(self):
        super(mynet,self).__init__()
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
        x=self.seq(x)
        return x
    
if __name__ == "__main__":
    model=mynet()
    print(model)
    input=torch.rand(64,3,32,32)
    output=model(input)
    print(output.shape)