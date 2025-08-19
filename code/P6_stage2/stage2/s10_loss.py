import torch
from torch import nn
from math import exp

input=torch.tensor([1,2,3],dtype=torch.float32).reshape(1,1,1,3)
target=torch.tensor([1,2,5],dtype=torch.float32).reshape(1,1,1,3)

loss_mse=nn.MSELoss()

result_mse=loss_mse(input,target)
print(result_mse)


x=torch.tensor([0.1,0.2,0.3]).reshape(1,3)
y=torch.tensor([1])

loss_cross=nn.CrossEntropyLoss()
result_cross=loss_cross(x,y)
print(result_cross)

z=exp(0.2)/(exp(0.1)+exp(0.2)+exp(0.3))
z=torch.tensor(z)
print(-torch.log(z))