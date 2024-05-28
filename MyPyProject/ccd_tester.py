import cv2
from PIL import Image
import numpy as np
import os
# import scipy.io
import glob
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from ccd_neural_net import My_CCD_NN
from ccd_my_dataset import My_CCD_Dataset, myselfTTNormalize
import torch.nn.functional as F


img_path = './result/CCD/image/'
img_list_fullname = sorted(glob.glob(img_path + '*.png'))
label_path = './result/CCD/labels568_ext012.npy'
lbs = np.load(label_path)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = My_CCD_NN(3)
model.to(device)
model_path = './models/ccd_myModel_remote.pth'
checkpoint = torch.load(model_path)
lowest_loss = checkpoint['lowest_loss']
model.load_state_dict(checkpoint['state_dict'])
print('Lowest Loss: {}'.format(lowest_loss))

# image = Image.fromarray(cv2.cvtColor(cv2.imread(img_list[0]), cv2.COLOR_BGR2RGB))


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


my_tf = transforms.Compose([
    transforms.Resize([224, 224]),  # 数据大小与训练集一致
    transforms.ToTensor(),
    # myselfTTNormalize(),
    # transforms.Normalize([0.11884392, 0.19811375, 0.10450162], [0.09571492, 0.1614415, 0.1012219]),
    # transforms.Normalize([0.051247064, 0.08537021, 0.048214838], [0.045985885, 0.07802741, 0.05130188])
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
ds = myDs(img_path, './result/CCD/labels568_ext012.npy', transform=my_tf)
img1, _ = ds[0]
print(img1.shape)
# 加载test
dl = DataLoader(ds, batch_size=16, shuffle=False)
# print(len(dl))

data_iter = iter(dl)
imgs, _ = data_iter.next()
imgs2, _ = data_iter.next()
imgs3, _ = data_iter.next()
imgs4, _ = data_iter.next()
imgs5, _ = data_iter.next()

model.eval()
output = model(imgs4.to(device))  # 0/2/4/5
# output = F.normalize(output, dim=1)  # tensor([-0.0301,  0.9993,  0.0241])
print(output[11], lbs[59], sep='\n')  # tensor([-0.0174,  0.5772,  0.0139])  [0.52995189 0.7187774  0.45001116]
print(imgs4[11])

tensor = imgs4[11].cpu().clone().squeeze(0)
tensor = tensor.permute(1, 2, 0)
image = tensor.numpy()
image = (image * 255).astype(np.uint8)
image = Image.fromarray(image)
# image.show()
# OpenCV 使用的是 BGR 通道顺序，因此需要将 RGB 转换为 BGR
image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
# img = cv2.imread(imgs[0], cv2.IMREAD_UNCHANGED)
# 实际效果
# cv2.imshow('ori1', np.array(image))
# cv2.waitKey(0)

# cv2.imshow('ori2', image_cv2)
# cv2.waitKey(0)

img = image_cv2
img = (img / ((2**12)-1)) * 100
img = np.clip(img * (255.0 / np.percentile(img, 100-2.5, keepdims=True)), 0, 255)
image = img.astype(np.uint8)
# image = np.transpose(Image.fromarray(image), (2, 0, 1))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)
print(image)

ground_truth = output[11].cpu().detach().numpy()  # lbs[0] [31] [59] [78]  output[0].cpu().detach().numpy()  [15]  [11]  [14]

Gain_R = float(np.max(ground_truth)) / float(ground_truth[0])
Gain_G = float(np.max(ground_truth)) / float(ground_truth[1])
Gain_B = float(np.max(ground_truth)) / float(ground_truth[2])

image[:, :, 0] = np.minimum(Gain_R * image[:, :, 0], 255)
image[:, :, 1] = np.minimum(Gain_G * image[:, :, 1], 255)
image[:, :, 2] = np.minimum(Gain_B * image[:, :, 2], 255)

# 白平衡操作
gamma = 1/2.2
image = pow(image, gamma) * (255.0 / pow(255, gamma))
image = np.array(image, dtype=np.uint8)
print(image.shape)
image8 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imwrite('./result/CCD/temp_res/' + 'res1.png', image8)
print('Success!')
img_res = cv2.imread('./result/CCD/temp_res/' + 'res1.png')
# cv2.imshow('res*1', img_res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
