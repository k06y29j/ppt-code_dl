import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter 

data_tran=transforms.Compose([  
    transforms.ToTensor()
    ])

train_set = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
)

test_set = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
)

writer=SummaryWriter(log_dir='./logs')

for i in range(10):
    img,target=test_set[i]
    writer.add_image('test_set',img,i)

writer.close()