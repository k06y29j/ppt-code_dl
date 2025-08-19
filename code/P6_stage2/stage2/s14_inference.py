import torch
import torchvision
from PIL import Image
from torch import nn

model_dict=torch.load("best_model.pth", weights_only=True)
model = torchvision.models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(p=0.2), 
    nn.Linear(512, 10)
)
model.load_state_dict(model_dict)

img_path="stage2/val_img/image2.jpg"
img=Image.open(img_path).convert('RGB').resize((32,32))

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
img = transform(img)
img = img.unsqueeze(0)
print(img.shape)

model.eval()
label_dict={0:"飞机",1:"汽车",2:"鸟",3:"猫",4:"鹿",5:"狗",6:"青蛙",7:"马",8:"船",9:"卡车"}

with torch.no_grad():
    output=model(img)
    print(output)
    inference_label=output.argmax(dim=1).item()
    print("预测结果为:{0},索引：{1}".format(label_dict[inference_label],inference_label))  