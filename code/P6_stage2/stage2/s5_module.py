from torch import nn
import torch.nn.functional as F

class myCNN(nn.Module):
    def __init__(self):
        super(myCNN,self).__init__()
        self.conv1=nn.Conv2d(64,64,3)
        self.conv2=nn.Conv2d(64,64,7,stride=2)

    def forward(self,x):
        x=F.relu(self.conv2(x))
        for i in range(3):
            x=F.relu(self.conv1(x))
        x=F.adaptive_avg_pool(x,2,2)
        return x