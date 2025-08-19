import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet18_Weights

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

dataset_train=torchvision.datasets.CIFAR10("../data",train=True,
                                      transform=train_transform,
                                      download=True)

dataloader_train=DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=4, pin_memory=True) 

data_test=torchvision.datasets.CIFAR10("../data",train=False,
                                      transform=test_transform,
                                      download=True)
dataloader_test=DataLoader(data_test, batch_size=128, num_workers=4, pin_memory=True)

# ResNet18
model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

model.fc = nn.Sequential(
    nn.Dropout(p=0.2), 
    nn.Linear(512, 10)
)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

#device=torch.device("cuda:1")
device=torch.device("cuda:2" if torch.cuda.is_available() and torch.cuda.device_count() > 2 else "cpu")
model=model.to(device)
loss=nn.CrossEntropyLoss().to(device)
writer=SummaryWriter("logs/logs_train")

train_step=0
test_step=0
epoch=10
best_test_loss = float('inf')
patience_counter = 0
patience = 5  
for i in range(epoch):
    print("第{}个epoch".format(i))
    loss_run=0.0
    correct = 0
    total = 0
    model.train()
    for data in dataloader_train:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss_value = loss(outputs, labels)
        
        # 计算准确率
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 反向传播和优化
        loss_run+=loss_value.item()
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        train_step+=1
        
    train_loss = loss_run/len(dataloader_train)
    accuracy = 100. * correct / total
    
    print("train_loss: ", train_loss)
    print("train_accuracy: {:.2f}%".format(accuracy))
    
    writer.add_scalar("train_loss", train_loss, i)
    writer.add_scalar("train_accuracy", accuracy, i)

    with torch.no_grad():
        model.eval()
        loss_run=0.0
        correct = 0
        total = 0
        for data in dataloader_test:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss_value = loss(outputs, labels)
            loss_run+=loss_value.item()
            
            # 计算准确率
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            test_step+=1
            
        test_loss = loss_run/len(dataloader_test)
        accuracy = 100. * correct / total
        
        print("test_loss: ", test_loss)
        print("test_accuracy: {:.2f}%".format(accuracy))
        
        # 更新学习率调度器
        scheduler.step(test_loss)
        
        writer.add_scalar("test_loss", test_loss, i)
        writer.add_scalar("test_accuracy", accuracy, i)
        
        # 保存最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"模型已保存! 最佳测试损失: {best_test_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 早停机制
        if patience_counter >= patience:
            print(f"早停! 连续{patience}个epoch没有改善")
            break
writer.close()

