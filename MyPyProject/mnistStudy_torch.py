# 加载数据集  minist数据集的格式有gz/npz/pkl.gz -gzip/zip  优化为自定义数据集,加载什么数据集怎么加载自己定
# CNN网络将邻近特征关联处理,更适合图像类的光源色彩分析
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import os
import gzip


# 定义超参数
input_size = 28  # 图像尺寸28*28*1是卷积网络的输入  类比全连接网络的features==784
num_classes = 10  # 标签种类数
num_epochs = 3  # 训练总循环周期
bs = 64

# 训练集与测试集 下载不下来MNIST/raw&processed
# train_ds = datasets.MNIST(root='./resource', train=True, transform=transforms.ToTensor(), download=True)
# test_ds = datasets.MNIST(root='./resource', train=False, transform=transforms.ToTensor())


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, folder, data_name, label_name, transform=None):
        (datas, labels) = self.load_data(folder, data_name, label_name)
        self.datas = datas
        self.labels = labels
        self.transform = transform


    def __getitem__(self, index):
        img, label = self.datas[index], self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


    def __len__(self):
        return len(self.datas)


    def load_data(self, folder, data_name, label_name):
        with gzip.open(os.path.join(folder, label_name), 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)

        with gzip.open(os.path.join(folder, data_name), 'rb') as imgpath:
            datas = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(labels), 28, 28)

        return (datas, labels)


train_ds = MyDataset('./resource/', 'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', transform=transforms.ToTensor())
test_ds = MyDataset('./resource/', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz', transform=transforms.ToTensor())
# 构建Batch加载数据
train_dl = DataLoader(dataset=train_ds, batch_size=bs, shuffle=True)
test_dl = DataLoader(dataset=test_ds, batch_size=bs, shuffle=True)


# 定义卷积网络
class Mnist_CNN(nn.Module):

    def __init__(self):
        super(Mnist_CNN, self).__init__()
        # 1:n卷积+池化+relu的组合封装进一个序列
        self.conv1 = nn.Sequential(  # 输入(1,28,28)  有时需要注意transport预处理读入的数据,因为可能格式是(28*28*1)
            nn.Conv2d(in_channels=1,  # 灰度图  3:RGB
                      out_channels=16,  # 要得到多少特征图
                      kernel_size=5,  # 卷积核/过滤器大小  1D针对结构化数据(不常用conv,更多fc),2D针对图像处理,3D针对视频-Conv3d
                      stride=1,  # 步长
                      padding=2),  # 在外面加2圈0;如果希望卷积后大小与原来一致,需要设置padding=(kernel_size-1)/2 if stride=1
            # 上面卷积的输出是(16,28,28)  [(h|w - ks + 2p)/s + 1):torch向下取整,不同框架可能不同
            nn.ReLU(),  # Relu层
            nn.MaxPool2d(kernel_size=2),  # 池化操作(2*2区域),输出(16,14,14), 压缩
        )
        self.conv2 = nn.Sequential(  # 输入(16,14,14)
            nn.Conv2d(16, 32, 5, 1, 2),  # 输出(32,14,14)
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出(32,7,7)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),  # 输出(64,7,7)
        )

        self.out = nn.Linear(64 * 7 * 7, 10)  # 全连接层得到结果

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # flatten操作, 结果: (batch_size, 64, 7, 7)  ~reshape
        out = self.out(x)
        return out


def accuracy(predictions, labels):
    pred = torch.max(predictions, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)


# 实例化
net = Mnist_CNN()
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
opt = optim.Adam(net.parameters(), lr=0.001)

# 循环训练
for epoch in range(num_epochs):
    train_rights = []  # 保存当前epoch结果

    for batch_idx, (data, target) in enumerate(train_dl):  # 针对容器中的每一个批循环
        net.train()
        output = net(data)
        loss = criterion(output, target)
        opt.zero_grad()
        loss.backward()
        opt.step()  # 参数更新
        right = accuracy(output, target)
        train_rights.append(right)

        if batch_idx % 100 == 0:  # 验证集不必跟随训练集每趟循环都跑,每隔100轮跑一下
            val_rights = []
            net.eval()
            for (data, target) in test_dl:
                # net.eval()
                output = net(data)
                right = accuracy(output, target)
                val_rights.append(right)

            # 准确率计算
            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
            valid_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

            print('当前Epoch: {} [{}/{} ({:.02f}%)]\t损失: {:.6f}\t训练集准确率: {:.2f}%\t测试集准确率: {:.2f}% '.format(
                epoch, batch_idx * bs, len(train_dl.dataset),
                100. * batch_idx / len(train_dl),
                loss.data,
                100. * train_r[0].numpy() / train_r[1],
                100. * valid_r[0].numpy() / valid_r[1],)
            )































# # FCN模板
# from pathlib import Path
# import requests
# import pickle
# import gzip
#
# from matplotlib import pyplot as plt
# import numpy as np
# import torch
# from torch import nn
# import torch.nn.functional as F  # 适用初步搭建模型的阶段测通
# from torch.utils.data import TensorDataset, DataLoader
#
#
# from torch import optim
#
#
# DATA_DIR = Path('./resource')
# path = DATA_DIR / 'TorchMinist'
# path.mkdir(parents=True, exist_ok=True)
# url = 'http://deeplearning.net/data/mnist/'
# file_name = 'mnist.pkl.gz'
# if not (path / file_name).exists():
#     content = requests.get(url + file_name).content
#     (path / file_name).open('wb').write(content)
#
# with gzip.open((path / file_name).as_posix(), 'rb') as f:  # pickle(除了最早的版本外)是二进制格式的,所以用'rb'标志打开文件
#     train_set, valid_set, test_set = pickle.load(f, encoding='latin-1')  # 训练集model.train()-验证集(训练后的考试)model.eval()-测试集
# train_x = train_set[0]
# train_y = train_set[1]
# valid_x = valid_set[0]
# valid_y = valid_set[1]
#
#
#
#
# # plt.imshow(train_x[0].reshape((28, 28)), cmap='gray')
# # plt.show()
# # print(train_x.shape)  # 50000,784=28*28*1  train_x[0].shape==784
# # print(train_y.shape)  # 50000 ()  train_y[:10] ~[] array
#
# # 转换numpy为tensor
# train_x, train_y, valid_x, valid_y = map(
#     torch.Tensor, (train_x, train_y, valid_x, valid_y)
# )
# dataTotalNum, features = train_x.shape  # 样本总数, 特征数(此处是像素点个数)
#
# # print(train_x, train_y)
# # print(train_x.shape)
# # print(train_y.min(), train_y.max())
#
#
# bs = 64  # 特征少,该值可以大一点
# steps = 20
# # x_batch = train_x[0:bs]  # xb
# # y_batch = train_y[0:bs]
# # weights = torch.randn([784, 10], dtype=torch.float, requires_grad=True)
# # bias = torch.zeros(10, requires_grad=True)
# #
# # def model(x_batch):
# #     return x_batch.mm(weights) + bias
# #
# loss_func = torch.nn.functional.cross_entropy
# # print(loss_func(model(x_batch), y_batch.long()))  # .item()
#
#
# class Mnist_NN(nn.Module):
#
#     def __init__(self):
#         # nn.Module的子类函数必须在构造函数中执行父类的构造函数,等价与nn.Module.__init__()
#         super(Mnist_NN, self).__init__()  # super().__init__()
#         # self.conv1 = nn.Conv2d(1, 32, 3, 1)  # (3*3*1)*32 1为输入通道数,32为输出通道数,3为卷积核(3维的就叫过滤器)大小,1为步长
#         # Conv2d返回的是一个Conv2d class的一个对象,该类中包含forward函数的实现,卷积运算图像尺寸变小了,原始是mxn的话,卷积核sxs,卷积后尺寸为:(m-s+1)x(n-s+1)
#         # self.flatten = nn.Flattern(默认start_dim=1, end_dim=-1)  # 维度展平为张量,卷积池化后降维与全连接层计算
#         # 假设conv2(32, 64, 3, 1), 另外池化2, 注意此时fc1输入是 3*3*32*64/2
#         self.hidden1 = nn.Linear(784, 128)  # fc1  weights和bias都会自动初始化
#         self.hidden2 = nn.Linear(128, 256)  # fc2
#         # 输出层
#         self.out = nn.Linear(256, 10)
#         # 处理过拟合,常用0.5  在前向传播的时候,让某个神经元的激活值以一定的概率p停止工作,增强模型泛化性,使它不会太依赖某些局部特征
#         self.dropout1 = nn.Dropout(0.5)
#         self.dropout2 = nn.Dropout(0.25)
#
#     # 前向传播需要人工定义,反向传播框架自动实现
#     def forward(self, x):  # x是一整个batch的输入内容
#         # 先卷积 x = F.relu(self.conv1) 可多层
#         # F.max_pool2d(x, 2) x为卷积核大小,2为步长
#         # 间歇插入self.dropout2(x)
#         # self.flatten(x, 1)之后可做全连接
#         x = torch.sigmoid(self.hidden1(x))  # 有全连接层时必做的两步 s1  F.relu() 激活函数强制稀疏处理,避免过拟合
#         x = self.dropout1(x)  # s2
#         x = torch.sigmoid(self.hidden2(x))
#         x = self.dropout2(x)
#
#         out = self.out(x)  # F.log_softmax(x, dim=)  # log取对数 dim-0:对每一列的所有元素进行softmax运算,并使得每一列所有元素和为1;dim-1:对每一行的所有元素进行softmax运算，并使得每一行所有元素和为1
#         return out
#
# # net = Mnist_NN()
# # print(net)
# # for name, param in net.named_parameters():
# #     print(name, param, param.size())
#
#
# train_ds = TensorDataset(train_x, train_y)
# # train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)  # 加载器
# valid_ds = TensorDataset(valid_x, valid_y)
# # valid_dl = DataLoader(valid_ds, batch_size=bs * 2)  # 验证集不需要严格按batch读入
#
#
# def get_data(train_ds, valid_ds, bs):
#     return (
#         DataLoader(train_ds, batch_size=bs, shuffle=True),
#         DataLoader(valid_ds, batch_size=bs * 2),
#     )
#
#
# def get_model():
#     model = Mnist_NN()
#     return model, optim.Adam(model.parameters(), lr=0.001)  # SGD在该模型下远逊于Adam
#
#
# train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
# net, opt = get_model()
#
#
# def loss_batch(model, loss_func, xb, yb, opt=None):  # 计算损失,更新参数
#     loss = loss_func(model(xb), yb.long())
#     loss.requires_grad_(True)  # loss默认的requires_grad是False,因此在backward()处不会计算梯度会导致出错
#
#     if opt is not None:
#         loss.backward()
#         opt.step()
#         opt.zero_grad()  # 每次迭代梯度计算是独立的
#
#     return loss.item(), len(xb)
#
#
# # 训练+验证
# def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
#     for epoch in range(epochs):
#         model.train()  # 更新权重w和偏置b
#         for xb, yb in train_dl:  # Batch step
#             loss_batch(model, loss_func, xb, yb, opt)
#
#         model.eval()  # 不更新 w b
#         with torch.no_grad():
#             losses, nums = zip(  # zip打包自行配对  zip*解包
#                 *[loss_batch(model, loss_func, xb, yb, opt) for xb, yb in valid_dl]
#             )
#         val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)  # 平均loss
#         print('当前Epoch: ' + str(epoch), '验证集损失: ' + str(val_loss))
#
#
# fit(steps, net, loss_func, opt, train_dl, valid_dl)
#
# # 计算验证集或测试集的准确率
# correct = 0
# total = 0
# for xb, yb in valid_dl:
#     outputs = net(xb)
#     _, predicted = torch.max(outputs.data, 1)  # 返回两个值,一个最大值(1-沿行-样本内部),另一个最大值所在位置(索引) _占位符
#     total += yb.size(0)
#     correct += (predicted == yb).sum().item()
#
# print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


