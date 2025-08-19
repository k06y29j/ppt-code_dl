from PIL import Image
from torchvision import transforms as T
import cv2

img_path = "/home/yongjia/pytorch/data/hymenoptera_data/train/ants/0013035.jpg"
"""
img为PIL类型，img1为numpy类型
"""
img=Image.open(img_path)
print(img)
img1=T.Resize((224, 224))(img)
img2=T.Resize(224)(img)
print(img1)
print(img2)
print(img1==img2)
# img1=cv2.imread(img_path)
# print(img1)
# img_tensor = T.ToTensor()(img)
# print(img_tensor)
# img_norm= T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
# print(img_norm)