import os
import numpy as np
import skimage
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class My_CCD_Dataset(Dataset):
    def __init__(self, data_dir, labels_path, flag='train', transform=None):
        assert flag in ['train', 'test']  # 区分训练集和测试集/验证集
        self.flag = flag
        self.imgs_dir = data_dir  # 未来考虑递归方案
        self.imgs = sorted(os.listdir(data_dir))
        self.lbs = np.load(labels_path)
        self.transform = transform
        self.data = self.splitData()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img = os.path.join(self.imgs_dir, self.data[index][0])
        lb = self.data[index][1]
        img = Image.open(img).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.transpose(np.array(img), (2, 0, 1))  # H*W*C -> C*H*W
        return img, lb

    def splitData(self):
        # 固定随机种子,打乱数据集顺序
        examples = list(zip(self.imgs, self.lbs))
        # 划分训练集和测试集
        train_data, test_data = train_test_split(examples, train_size=0.85, test_size=0.15, shuffle=True, random_state=42)  # stratify
        if self.flag == 'train':
            split_data = train_data
        else:
            split_data = test_data
        return split_data


class myselfTTNormalize(object):  # 替代ToTensor对RGB等Image转换成tensor,并修改(粗暴/255)归一化的工作
    def __call__(self, image):
        img = image
        # 将图像张量转换为 double/float64 类型->float32 .astype(np.float32)  img.float32()
        img_val = np.transpose(skimage.img_as_float(img).astype(np.float32), (2, 0, 1))
        # 计算张量的最小值和最大值
        img_val = torch.tensor(img_val)  # ToTensor()但未粗暴除以255归一化
        min_val = torch.min(img_val)
        max_val = torch.max(img_val)
        # 对图像进行归一化处理[0,1]
        # 图像归一化使得图像可以抵抗几何变换的攻击,它能够找出图像中的那些不变量,从而得知这些图像原本就是一样的或者一个系列的
        normalized_image = (img_val - min_val) / (max_val - min_val)
        return normalized_image


data_dir = './result/CCD/image/'
labels_path = './result/CCD/labels568_ext012.npy'
my_tf = {
    'train':
        transforms.Compose([
            # 数据增强,使数据具备多样性
            transforms.Resize([96, 96]),  # 3*224*224
            transforms.RandomRotation(45),
            transforms.CenterCrop(64),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),  # ColorJitter极端光照条件调节
            myselfTTNormalize(),
            # transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # (mean,std)  将数据标准化,即均值为0,标准差为1,使模型更容易收敛
            # transforms.Normalize([0.11884392, 0.19811375, 0.10450162], [0.09571492, 0.1614415, 0.1012219])
        ]),
    'test':
        transforms.Compose([
            transforms.Resize([64, 64]),  # 数据大小与训练集一致
            myselfTTNormalize(),  # transforms.ToTensor()
            # transforms.Normalize([0.11884392, 0.19811375, 0.10450162], [0.09571492, 0.1614415, 0.1012219])
        ]),
}

ds = {x: My_CCD_Dataset(data_dir, labels_path, flag=x, transform=my_tf[x]) for x in ['train', 'test']}
# 加载数据集
train_dl = DataLoader(ds['train'], batch_size=16, shuffle=True)
test_dl = DataLoader(ds['test'], batch_size=16)
dl = {'train': train_dl, 'test': test_dl}

# print(ds['train'].__len__())
# print(ds['test'].__len__())
# img, lb = ds['train'][0]
# img1, lb1 = ds['test'][0]
# print(img.shape, lb.shape)
# print(img1.shape, lb1.shape)
# print(img1.max(), img1.min())
# print(ds['train'][0])


# 计算mean和std
def getStat(train_data):
    print(len(train_data))
    train_loader = DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


# mean, std = getStat(ds['train'])
# print(mean, std)


