import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image  # plt.imshow()中,接受图片类型np.ndarray/tensor/PIL Image这些任意的类型
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torchvision.models import squeezenet1_1, resnet18, resnet50
import sys
import copy
import random
import time
# import imageio
# from skimage import io
import warnings
warnings.filterwarnings("ignore")
# from sklearn.model_selection import train_test_split


img_dir = './trying/old/'
lb_full_name = './trying/labels568_ext012.npy'


class myDs(Dataset):
    def __init__(self, img_dir, lb_full_name, transform=None):
        self.img_dir = img_dir
        self.imgs = sorted(os.listdir(img_dir))
        self.lbs = np.load(lb_full_name)
        self.transform = transform


    def __getitem__(self, index):
        img = os.path.join(self.img_dir, self.imgs[index])
        img = Image.open(img).convert("RGB")
        lb = self.lbs[index]
        # lb = torch.Tensor(lb)
        if self.transform is not None:
            img = self.transform(img)
        return img, lb


    def __len__(self):
        return len(self.imgs)


# 自定义转换函数
class myNormalize(object):
    def __call__(self, img):
        image = img
        # 将图像张量转换为 float32 类型
        img_val = image.float()
        # 计算张量的最小值和最大值
        min_val = torch.min(img_val)
        max_val = torch.max(img_val)
        # 对图像进行归一化处理
        normalized_image = (img_val - min_val) / (max_val - min_val)
        return normalized_image


bs = 16
epochs = 20

myTransform = transforms.Compose([
    # transforms.Resize((96, 96)),  # 调整图像大小224  96/64
    transforms.ToTensor(),  # 转换为Tensor
    myNormalize(),
])

ds = myDs(img_dir, lb_full_name, transform=myTransform)  # ds[0]

train_nums = int(0.85 * len(ds))  # 1448  1704*0.85=1448.4
test_nums = len(ds) - train_nums  # 256
train_ds, test_ds = torch.utils.data.random_split(ds, [train_nums, test_nums])
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=bs)
dl = {'train': train_dl, 'valid': test_dl}


model_name = 'resnet'
feature_extract = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model_ft = resnet18()
# print(model_ft)


def set_param_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False  # 冻结


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = model_name(pretrained=use_pretrained)
    set_param_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft


model_ft = initialize_model(resnet18, 3, feature_extract, use_pretrained=True)  # /home/bian/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
model_ft.to(device)
print(model_ft)
filename = './models/try_local.pt'  # pt/pth
params_to_update = model_ft.parameters()
print('Parameters to learn: ')
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print('\t', name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print('\t', name)

optimizer = optim.Adam(params_to_update, lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
criterion = nn.CrossEntropyLoss()


def train(model, dl, loss, opt, epochs, filename):
    since = time.time()
    model.to(device)
    lowest_loss = 1e10
    train_losses = []
    valid_losses = []
    LRs = [opt.param_groups[0]['lr']]
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs-1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels in dl[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                opt.zero_grad()
                outputs = model(inputs)
                cost = loss(outputs, labels)
                if phase == 'train':
                    cost.backward()
                    opt.step()

                running_loss += cost.item() * inputs.size(0)

            epoch_loss = running_loss / len(dl[phase].dataset)

            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'valid' and epoch_loss < lowest_loss:
                lowest_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'lowest_loss': lowest_loss,
                    'optimizer': opt.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'valid':
                valid_losses.append(epoch_loss)
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
    return model, valid_losses, train_losses, LRs


model_1, valid_losses, train_losses, LRs = train(model_ft, dl, criterion, optimizer, epochs, filename)

# 重新训练所有层
for param in model_1.parameters():
    param.requires_grad = True

optimizer2 = optim.Adam(model_1.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer2, step_size=5, gamma=0.1)

checkpoint = torch.load(filename)
# lowest_loss = checkpoint['lowest_loss']
model_1.load_state_dict(checkpoint['state_dict'])

model_1, valid_losses, train_losses, LRs = train(model_1, dl, criterion, optimizer2, epochs, filename)



# test
model_2 = initialize_model(resnet18, 3, feature_extract, use_pretrained=True)
model_2 = model_2.to(device)
checkpoint2 = torch.load(filename)
lowest_loss = checkpoint2['lowest_loss']
model_2.load_state_dict(checkpoint2['state_dict'])

data_iter = iter(dl['valid'])
imgs, lbs = data_iter.next()

model_2.eval()
output = model_2(imgs.cuda())  # GPU: imgs.cuda()
print(output, lbs.data, sep='\n')


