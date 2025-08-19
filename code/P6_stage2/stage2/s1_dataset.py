"""Dataset(数据，编号，标签), Dataloader(对dataset的封装，提供batch数据)"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

root_dir="/home/yongjia/pytorch/data/hymenoptera_data/train"
ant_label_dir="ants"
bee_label_dir="bees"

class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path=os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)
        

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_path = os.path.join(self.path, img_name)
        img = Image.open(img_path)
        label=self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

ant_dataset = MyData(root_dir, ant_label_dir)
bee_dataset=MyData(root_dir, bee_label_dir)

train_dataset = ant_dataset + bee_dataset
print(train_dataset[129])  

