import numpy as np
from matplotlib import pyplot as plt  # 画图
import torch
import torch.nn as nn
import random
import time

# 随机种子,实验可重复性
seed_id = 1234
random.seed(seed_id)
np.random.seed(seed_id)
torch.manual_seed(seed_id)
torch.cuda.manual_seed(seed_id)
torch.cuda.manual_seed_all(seed_id)

# 构建简易数据集,以sin函数为例
x = np.linspace(-np.pi, np.pi).astype(np.float32)
y = np.sin(x)
x_train = random.sample(x.tolist(), 25)  # 输入网络的data,取25个点
y_train = np.sin(x_train)  # 相当于输入data的label

plt.scatter(x_train, y_train, c="r")
plt.plot(x, y)
plt.show()

# 利用pytorch框架搭建神经网络模型
class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [1, 20, 1]
        self.layer1 = nn.Linear(layers[0], layers[1])
        self.layer2 = nn.Linear(layers[1], layers[2])
        self.elu = nn.ELU()
    def forward(self, d):
        d1 = self.layer1(d)
        d1 = self.elu(d1)
        d2 = self.layer2(d1)
        return d2

# 基础设定  明确指定cuda:0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 10000  # 迭代次数
learning_rate = 1e-4  # 学习率
net = DNN().to(device=device)  # 网络初始化
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # 优化器,这里使用随机梯度下降SGD的一种类型Adam自适应优化器
mse_loss = nn.MSELoss()  # 损失函数,这里使用MSE均方误差损失函数，用于衡量模型输出与实际标签之间的差异
min_train_loss = 1e10
train_loss = []
pt_x_train = torch.from_numpy(np.array(x_train)).to(device=device, dtype=torch.float32).reshape(-1, 1)
pt_y_train = torch.from_numpy(np.array(y_train)).to(device=device, dtype=torch.float32).reshape(-1, 1)
print(pt_x_train.dtype)
print(pt_x_train.shape)

# 网络训练过程
start_0 = time.time()
start = time.time()
end = time.time()
for epoch in range(1, epochs+1):
    net.train()  # 更新网络参数
    pt_y_pred = net(pt_x_train)
    loss = mse_loss(pt_y_pred, pt_y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        end = time.time()
        print("epoch:[%5d/%5d] time:%.2fs current_loss:%.5f"
              %(epoch, epochs, (end-start), loss.item()))
        start = time.time()
    train_loss.append(loss.item())
    if train_loss[-1] < min_train_loss:
        torch.save(net.state_dict(), "./models/sinModel.pth")  # 保存每一次loss下降的模型
        min_train_loss = train_loss[-1]
end_0 = time.time()
print("训练总用时:%.2fmin"
      %((end_0-start_0) / 60))
# loss收敛情况
plt.plot(range(epochs), train_loss)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# 导入训练好的模型,输入验证数据,预测结果
x_test = np.linspace(-np.pi, np.pi).astype(np.float32)
pt_x_test = torch.from_numpy(x_test).to(device=device,dtype=torch.float32).reshape(-1, 1)
dnn = DNN().to(device)
dnn.load_state_dict(torch.load("./models/sinModel.pth", map_location=device))  # pytorch导入模型
dnn.eval()  # 仅用于评价模型,不反传
pt_y_test = dnn(pt_x_test)
# 可视化
y_test = pt_y_test.detach().cpu().numpy()  # 输出结果torch tensor,需要转化为numpy类型
plt.scatter(x_train, y_train, c="r")
plt.plot(x_test, y_test)
plt.show()















