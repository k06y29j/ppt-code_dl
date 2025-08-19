"""读取和存储"""
import torch
from torch import nn
from torch.nn import functional as F
import time

x=torch.arange(4)
torch.save(x,"x-file")

x2=torch.load("x-file", weights_only=True)
print(x2)

y=torch.zeros(4)
#[x,y] 表示将x和y打包成一个列表，然后保存到xy-file文件中
torch.save([x,y],"xy-file")
x2,y2=torch.load("xy-file", weights_only=True)
print(x2,y2)

mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict', weights_only=True)
print(mydict2)

print("---------加载和保存模型参数-----------------")
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(20,256)
        self.output=nn.Linear(256,10)
    def forward(self,x):
        return self.output(F.relu(self.hidden(x)))
net=MLP()
X=torch.randn(2,20)
Y=net(X)
torch.save(net.state_dict(),'mlp.params')

clone=MLP()
clone.load_state_dict(torch.load('mlp.params', weights_only=True))
# 设置为评估模式，评估模式下，模型不会进行训练，不会更新参数
clone.eval()
Y_clone = clone(X)
print(Y_clone == Y)
print(torch.cuda.device_count())

print("---------GPU性能优化示例-----------------")

# 检查是否有GPU可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建模型和数据
model = MLP().to(device)
X = torch.randn(100, 20).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ❌ 错误做法：频繁的GPU-CPU数据传输
print("❌ 错误做法 - 频繁GPU-CPU传输:")
for epoch in range(5):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, torch.randn_like(output))
    loss.backward()
    optimizer.step()
    
    # 每次打印都会触发GPU-CPU传输，性能很差
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")  # loss.item() 触发传输

# ✅ 正确做法1：批量收集损失，最后一次性传输
print("\n✅ 正确做法1 - 批量收集损失:")
losses = []
for epoch in range(5):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, torch.randn_like(output))
    loss.backward()
    optimizer.step()
    
    # 将损失存储在GPU内存中
    losses.append(loss.detach())  # 不触发传输

# 最后一次性传输所有损失
losses_cpu = [loss.item() for loss in losses]
for epoch, loss in enumerate(losses_cpu):
    print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ✅ 正确做法2：使用torch.no_grad()减少内存使用
print("\n✅ 正确做法2 - 使用no_grad():")
for epoch in range(5):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, torch.randn_like(output))
    loss.backward()
    optimizer.step()
    
    # 在no_grad上下文中，item()操作更高效
    with torch.no_grad():
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print("\n性能优化总结:")
print("1. 避免频繁的GPU-CPU数据传输")
print("2. 批量收集数据后一次性传输")
print("3. 使用torch.no_grad()减少内存使用")
print("4. 使用专门的日志工具如TensorBoard")
print("5. 考虑使用混合精度训练")
print("6. 减少打印频率，只在关键点打印")

print("\n---------矩阵乘法时间测量示例-----------------")

# 设置矩阵大小和数量
matrix_size = 100
num_matrices = 1000

print(f"测量 {num_matrices} 个 {matrix_size}x{matrix_size} 矩阵乘法的时间")
print(f"使用设备: {device}")

# ❌ 方法1：逐个记录结果（性能较差）
print("\n❌ 方法1：逐个记录结果（性能较差）")
start_time = time.time()

norms_individual = []
for i in range(num_matrices):
    # 创建随机矩阵
    A = torch.randn(matrix_size, matrix_size, device=device)
    B = torch.randn(matrix_size, matrix_size, device=device)
    
    # 矩阵乘法
    C = torch.mm(A, B)
    
    # 计算Frobenius范数并立即传输到CPU（性能瓶颈）
    norm = torch.norm(C, p='fro').item()  # 每次item()都触发GPU-CPU传输
    norms_individual.append(norm)
    
    # 每100次打印一次进度
    if (i + 1) % 100 == 0:
        print(f"完成 {i + 1}/{num_matrices} 个矩阵")

end_time = time.time()
individual_time = end_time - start_time
print(f"逐个记录方法耗时: {individual_time:.4f} 秒")
print(f"平均每个矩阵耗时: {individual_time/num_matrices*1000:.2f} 毫秒")

# ✅ 方法2：批量收集后传输（性能较好）
print("\n✅ 方法2：批量收集后传输（性能较好）")
start_time = time.time()

norms_batch = []
for i in range(num_matrices):
    # 创建随机矩阵
    A = torch.randn(matrix_size, matrix_size, device=device)
    B = torch.randn(matrix_size, matrix_size, device=device)
    
    # 矩阵乘法
    C = torch.mm(A, B)
    
    # 计算Frobenius范数但保持在GPU上
    norm = torch.norm(C, p='fro').detach()  # 不触发传输
    norms_batch.append(norm)
    
    # 每100次打印一次进度
    if (i + 1) % 100 == 0:
        print(f"完成 {i + 1}/{num_matrices} 个矩阵")

# 最后一次性传输所有结果
norms_batch_cpu = [norm.item() for norm in norms_batch]

end_time = time.time()
batch_time = end_time - start_time
print(f"批量收集方法耗时: {batch_time:.4f} 秒")
print(f"平均每个矩阵耗时: {batch_time/num_matrices*1000:.2f} 毫秒")

# 性能对比
speedup = individual_time / batch_time
print(f"\n性能对比:")
print(f"逐个记录方法: {individual_time:.4f} 秒")
print(f"批量收集方法: {batch_time:.4f} 秒")
print(f"性能提升: {speedup:.2f}x")

# 验证结果一致性
print(f"\n结果验证:")
print(f"两种方法计算的范数是否一致: {torch.allclose(torch.tensor(norms_individual), torch.tensor(norms_batch_cpu))}")
print(f"范数范围: {min(norms_individual):.4f} - {max(norms_individual):.4f}")

# 显示前几个结果作为示例
print(f"\n前5个矩阵的Frobenius范数:")
for i in range(5):
    print(f"矩阵 {i+1}: {norms_individual[i]:.4f}")