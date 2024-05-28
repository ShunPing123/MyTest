import os
import numpy as np
from PIL import Image  # plt.imshow()中,接受图片类型np.ndarray/tensor/PIL Image这些任意的类型
from matplotlib import pyplot as plt
# from skimage import io
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import copy
import time
from ccd_my_dataset import My_CCD_Dataset, myselfTTNormalize
from ccd_neural_net import My_CCD_NN


# 定义超参数
outputs = 3
bs = 16
# steps = 10000  # epochs = steps / ((dataTotalNum/bs)]  10000/(1704/16)
epochs1 = 25
epochs2 = 40
learning_rate = 0.001


# 加载数据集
data_dir = './result/CCD/image/'
labels_path = './result/CCD/labels568_ext012.npy'
my_tf = {
    'train':
        transforms.Compose([
            # 数据增强,使数据具备多样性
            transforms.Resize([256, 256]),  # 3*224*224  96, 96
            transforms.RandomRotation(45),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),  # ColorJitter极端光照条件调节
            myselfTTNormalize(),  # transforms.ToTensor()
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # (mean,std)  将数据标准化,即均值为0,标准差为1,使模型更容易收敛
            # transforms.Normalize([0.11884392, 0.19811375, 0.10450162], [0.09571492, 0.1614415, 0.1012219])
        ]),
    'test':
        transforms.Compose([
            transforms.Resize([224, 224]),  # 数据大小与训练集一致
            myselfTTNormalize(),  # transforms.ToTensor()
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.11884392, 0.19811375, 0.10450162], [0.09571492, 0.1614415, 0.1012219])
        ]),
}

ds = {x: My_CCD_Dataset(data_dir, labels_path, flag=x, transform=my_tf[x]) for x in ['train', 'test']}
# 加载
train_dl = DataLoader(ds['train'], batch_size=bs, shuffle=True)
test_dl = DataLoader(ds['test'], batch_size=bs)
dl = {'train': train_dl, 'test': test_dl}


# 创建模型实例  w,b自动初始化
model_ft = My_CCD_NN(outputs)


# 定义损失函数和优化器
def angle_losses(prediction, _labels):
    safe_v = torch.tensor(0.999999)

    predicting_illuminant = prediction
    standard_illuminant = _labels

    dot = torch.sum(predicting_illuminant * standard_illuminant, dim=1)
    dot = torch.clamp(dot, -safe_v, safe_v)

    angle = torch.acos(dot) * (180 / np.pi)
    angle_mean = torch.mean(angle)

    # 将标量值封装在一个 PyTorch 张量中
    angle_tensor = angle_mean.unsqueeze(0)

    return angle_tensor


# loss_func = nn.CrossEntropyLoss()  # F.cross_entropy
opt1 = optim.Adam(model_ft.parameters(), lr=learning_rate)
# 衰减
scheduler1 = optim.lr_scheduler.StepLR(opt1, step_size=7, gamma=0.1)
# 设备选择
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft.to(device)

feature_extract = True
params_to_update = model_ft.parameters()
# print('Parameters to learn: ')
# if feature_extract:
#     params_to_update = []
#     for name, param in model_ft.named_parameters():
#         if param.requires_grad == True:
#             params_to_update.append(param)
#             print('\t', name)
# else:
#     for name, param in model_ft.named_parameters():
#         if param.requires_grad == True:
#             print('\t', name)

filename = './models/ccd_myModel_local.pth'


# 训练
def train(model, dl, epochs, opt, scheduler, filename):  # , loss
    since = time.time()
    model.to(device)
    lowest_loss = 1e10
    train_losses = []
    test_losses = []
    LRs = [opt.param_groups[0]['lr']]
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs+1):
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels in dl[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                opt.zero_grad()
                outputs = model(inputs)
                # print('·' * 10)
                # print(outputs)
                # print('·' * 10)
                cost = angle_losses(outputs, labels)
                if phase == 'train':
                    cost.backward()
                    opt.step()

                running_loss += cost.item() * inputs.size(0)

            epoch_loss = running_loss / len(dl[phase].dataset)

            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'test' and epoch_loss < lowest_loss:
                lowest_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'lowest_loss': lowest_loss,
                    'optimizer': opt.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'test':
                test_losses.append(epoch_loss)
                # scheduler.step(epoch_loss)
            if phase == 'train':
                train_losses.append(epoch_loss)

        print('Optimizer learning rate: {:.7f}'.format(opt.param_groups[0]['lr']))
        LRs.append(opt.param_groups[0]['lr'])
        print()
        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(lowest_loss))

    model.load_state_dict(best_model_wts)
    return model, test_losses, train_losses, LRs


model_1, test_losses, train_losses, LRs = train(model_ft, dl, epochs1, opt1, scheduler1, filename)
# 重新训练所有层
for param in model_1.parameters():
    param.requires_grad = True
opt2 = optim.Adam(model_1.parameters(), lr=1e-4)
scheduler2 = optim.lr_scheduler.StepLR(opt2, step_size=10, gamma=0.1)
model_2, test_losses, train_losses, LRs = train(model_1, dl, epochs2, opt2, scheduler2, filename)



# # test
# import cv2
# import numpy as np
# # import scipy.io
# import glob
#
# img_path = './trying/old/'
# img_list = sorted(glob.glob(img_path + '*.png'))
# print(len(img_list))
# lbs = np.load('./result/CCD/labels568_ext012.npy')
#
# for i in range(7):
#     img = cv2.imread(img_list[i], cv2.IMREAD_UNCHANGED)
#
#     img = (img / ((2**12)-1)) * 100
#     img = np.clip(img * (255.0 / np.percentile(img, 100-2.5, keepdims=True)), 0, 255)
#     image = img.astype(np.uint8)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     ground_truth = lbs[i, :]
#
#     Gain_R = float(np.max(ground_truth)) / float(ground_truth[0])
#     Gain_G = float(np.max(ground_truth)) / float(ground_truth[1])
#     Gain_B = float(np.max(ground_truth)) / float(ground_truth[2])
#
#     image[:, :, 0] = np.minimum(Gain_R * image[:, :, 0], 255)
#     image[:, :, 1] = np.minimum(Gain_G * image[:, :, 1], 255)
#     image[:, :, 2] = np.minimum(Gain_B * image[:, :, 2], 255)
#
#     # 白平衡操作
#     gamma = 1/2.2
#     image = pow(image, gamma) * (255.0 / pow(255, gamma))
#     image = np.array(image, dtype=np.uint8)
#
#     image8 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     cv2.imwrite('./trying/new/' + str(i+1) + '.png', image8)
#
#     # if i % 10 == 0:
#     print(str(i+1) + ' Success!')
