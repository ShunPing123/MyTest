# 预测气温-回归预测任务  此前minist属于分类任务
import warnings
warnings.filterwarnings('ignore')
import numpy as np  # 矩阵计算
import pandas as pd  # 数据基本处理
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import datetime  # 处理时间数据
from sklearn import preprocessing

# %matplotlib inline  用于jupyterQtconsole/Notebook内嵌画图

# 查看数据的样子 pd的head方法默认读取前五行
features = pd.read_csv('./resource/TEMPS/temps.csv')
print(features.head())
print('数据维度:', features.shape)  # 348 9
# 取出年月日
years = features['year']
months = features['month']
days = features['day']
# datetime标准时间格式,便于随时间画趋势
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# zip:
# dates[:5]

# 画图,默认风格
plt.style.use('fivethirtyeight')
# 设置子图布局 画图呈现的效果
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
fig.autofmt_xdate(rotation=45)  # x轴倾斜45
# 标签 实际值
ax1.plot(dates, features['actual'])
ax1.set_xlabel('')
ax1.set_ylabel('Temperature')
ax1.set_title('Max Temp')
# 昨天
ax2.plot(dates, features['temp_1'])
ax2.set_xlabel('')
ax2.set_ylabel('Temperature')
ax2.set_title('Previous Max Temp')
# 前天
ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date')
ax3.set_ylabel('Temperature')
ax3.set_title('Two Days Prior Max Temp')
# 我的好朋友  偏差较大,可以做数据清洗 .drop .to_csv("new.csv", index=False, encoding="utf-8")
# 优化数据集对训练结果影响不大,更好的方式是优化模型
ax4.plot(dates, features['friend'])
ax4.set_xlabel('Date')
ax4.set_ylabel('Temperature')
ax4.set_title('Friend Estimate')  # 朋友估计

plt.tight_layout(pad=2)  # 子图间隔pad
# 画图不显示,可能原因:多个库有冲突,加上下面这行代码可以指定显示画图
matplotlib.pyplot.show()

# week是字符串型,神经网络无法直接处理,须先转化为数值  one-hot独热编码:在对应星期数值为1,其余为0
features = pd.get_dummies(features)  # pd自动区分字符串不重复的属性值进行独热编码
print(features.head(7))

# 标签提取
labels = np.array(features['actual'])
# 在特征中去掉标签列
features = features.drop('actual', axis=1)
# 每列的名字单独保存一下，以备后患
feature_list = list(features.columns)
# np.array()把列表转化为数组
features = np.array(features)
print(features.shape)

# 不仅计算训练数据的均值和方差，还会基于计算出来的均值和方差来转换训练数据，从而把数据转换成标准的正太分布
input_features = preprocessing.StandardScaler().fit_transform(features)  # (x-u)/o
print(input_features[0])



# 构建网络模型
# version2.0 适用GPU改写
input_size = input_features.shape[1]  # 特征个数14
hidden_size = 128  # 隐含层神经元个数
hidden2_size = 256  # 隐含层神经元个数
output_size = 1
batch_size = 32  # MINI-Batch:全部样本内存放不下  在该模型中32比16平均损失大
steps = 2000
losses = []
my_nn = torch.nn.Sequential(  # 主要用于预训练模型
    # 顺序执行以下步骤
    torch.nn.Linear(input_size, hidden_size),  # 全连接层 权重参数自动随机初始化
    torch.nn.Sigmoid(),  # 激活函数  nn.ReLU()  Tanh()适用于较浅,Relu()适合较深;Tanh和Sigmoid适合非线性;激活函数可以通过卷积层运算来代替
    torch.nn.Linear(hidden_size, hidden2_size),
    torch.nn.ReLU(),  # 此处换用Relu拟合度更好一点
    torch.nn.Linear(hidden2_size, output_size),
)
# MSE损失函数  此处不要反余弦/交叉熵损失函数-Tensor需要Long:交叉熵loss对于one-hot又会进行升维是报错的根源
# cost = torch.nn.MSELoss(reduction='mean')  # mean默认 sum none  L1Loss()
cost = torch.nn.SmoothL1Loss()  # nn.CrossEntropyLoss()~nn.LogSoftmax()+nn.NLLLoss()
# 优化器-更新参数  SGD/SGDM  Adam动量:惯性
optimizer = optim.Adam(my_nn.parameters(), lr=0.001)
# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
my_nn.to(device)
# 将input_features和labels移动到GPU上
input_features = torch.tensor(input_features, dtype=torch.float).to(device)
labels = torch.tensor(labels, dtype=torch.float).to(device)
# 训练
for i in range(steps):
    batch_loss = []
    for start in range(0, len(input_features), batch_size):
        # 越界处理
        end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
        xx = torch.tensor(input_features[start:end],
                          dtype=torch.float,
                          requires_grad=True)
        yy = torch.tensor(labels[start:end],
                          dtype=torch.float,
                          requires_grad=True)
        prediction = my_nn(xx)  # 前向传播
        loss = cost(prediction, yy)  # 计算损失
        optimizer.zero_grad()  # 梯度清零
        loss.backward(retain_graph=True)  # 反向传播
        optimizer.step()  # 更新参数
        batch_loss.append(loss.data.cpu().numpy())  # 记录损失，便于打印
    # 打印损失
    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))




# # version1.0
# x = torch.tensor(input_features, dtype=float)
# y = torch.tensor(labels, dtype=float)
# # 权重参数初始化，构建2层神经网络 （只有1个隐含层）
# # [348,14]->[14,128]->[128,1]（输入层->隐含层->输出层）
# # 第1层有128个神经元
# # randn，从标准正态分布中返回⼀个或多个样本值
# weights = torch.randn((14, 128), dtype=float, requires_grad=True)  # 权值矩阵w
# biases = torch.randn(128, dtype=float, requires_grad=True)  # 偏置向量b
# # 第2层有1个神经元
# weights2 = torch.randn((128, 1), dtype=float, requires_grad=True)
# biases2 = torch.randn(1, dtype=float, requires_grad=True)
#
# learning_rate = 0.001  # 学习率小一点好,梯度更新幅度的方向走多远
# steps = 2000
# losses = []
#
# for i in range(steps):
#     # 计算隐层  w*x+b
#     hidden = x.mm(weights) + biases
#     # 加入激活函数 relu sigmoid
#     hidden = torch.relu(hidden)
#     # 预测结果
#     predictions = hidden.mm(weights2) + biases2
#     # 全样本通计算损失
#     loss = torch.mean((predictions - y) ** 2)  # 均方差 MSE
#     losses.append(loss.data.numpy())  # tensor转数组格式
#     # 打印损失值
#     if i % 100 == 0:
#         print('loss:', loss)
#
#     # 返向传播计算
#     loss.backward()
#     # 更新参数
#     weights.data.add_(-learning_rate * weights.grad.data)
#     biases.data.add_(-learning_rate * biases.grad.data)
#     weights2.data.add_(-learning_rate * weights2.grad.data)
#     biases2.data.add_(-learning_rate * biases2.grad.data)
#
#     # 每次迭代都得记得清空 torch和tensorflow的区别点,torch需要清零否则会累加
#     weights.grad.data.zero_()
#     biases.grad.data.zero_()
#     weights2.grad.data.zero_()
#     biases2.grad.data.zero_()



# 预测训练结果
# 再次载入全部样本
x = torch.tensor(input_features, dtype=torch.float)
# 预测结果转换成numpy格式便于画图
predict = my_nn(x).data.numpy()
print(predict)
# 创建一个表格存放日期和对应的标签
true_data = pd.DataFrame(data={'date': dates, 'actual': labels})
# 同理,再创建一个来存日期和其对应的模型预测值
years = features[:, feature_list.index('year')]
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
test_dates = [
    str(int(year)) + '-' + str(int(month)) + '-' + str(int(day))
    for year, month, day in zip(years, months, days)
]
test_dates = [
    datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates
]

predictions_data = pd.DataFrame(data={
    'date': test_dates,
    'prediction': predict.reshape(-1)  # 变成1串,没有行列  1列:(-1, 1) 1行:(1, -1)
})

# 绘制图形准备
plt.figure(figsize=(20, 10))
plt.gcf().subplots_adjust(left=0.05,top=0.91,bottom=0.15)  # 显示不全
# 真实值
plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')
# 预测值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
plt.xticks(rotation='60')
plt.legend()  # 在图上添加标识
# 图名
plt.xlabel('Date')
plt.ylabel('Maximum Temperature (F)')
plt.title('Actual and Predicted Values')
plt.show()


