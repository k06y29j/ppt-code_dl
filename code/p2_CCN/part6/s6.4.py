"""多输入通道和多输出通道"""
import torch
from d2l import torch as d2l

def corr2d_multi_in(X,K):
    #
    return sum(d2l.corr2d(x,k) for x,k in zip(X,K)) # 对每个通道进行互相关运算，然后求和

X=torch.tensor([[[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]],
                [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]])
K=torch.tensor([[[0.0,1.0],[2.0,3.0]],[[1.0,2.0],[3.0,4.0]]])
print(X.shape,"/n",K.shape)
print(corr2d_multi_in(X,K))

def corr2d_multi_in_out(X,K):
    #torch.stack 在新的维度上连接张量
    return torch.stack([corr2d_multi_in(X,k) for k in K],0)
# 将K复制3次，形成一个3通道的卷积核,0表示在第0维度上复制
K=torch.stack((K,K+1,K+2),0)
print(K.shape)
print(corr2d_multi_in_out(X,K))

print("--------------1*1卷积核------------------")
def corr2d_multi_in_out_1x1(X,K):
    c_i,h,w=X.shape
    c_o=K.shape[0]
    X=X.reshape((c_i,h*w))
    K=K.reshape((c_o,c_i))
    Y=torch.matmul(K,X)
    return Y.reshape(c_o,h,w)

X=torch.normal(0,1,(3,3,3))
K=torch.normal(0,1,(2,3,1,1))

Y1=corr2d_multi_in_out_1x1(X,K)
Y2=corr2d_multi_in_out(X,K)
assert float(torch.abs(Y1-Y2).sum())<1e-6
print("测试通过")