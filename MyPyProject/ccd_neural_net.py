import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import squeezenet1_1, resnet18, resnet50

class My_CCD_NN(nn.Module):
    def __init__(self, outputs, feature_extract=True, use_pretrained=True):
        super(My_CCD_NN, self).__init__()
        self.squeezenet = squeezenet1_1(pretrained=use_pretrained)
        self.resnet = resnet18(pretrained=use_pretrained)
        # 冻结模型的参数
        self.set_param_requires_grad(self.squeezenet, feature_extract)
        self.set_param_requires_grad(self.resnet, feature_extract)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = nn.Linear(32*32, 536)
        self.dropout = nn.Dropout(0.5)
        num_ftrs = self.resnet.fc.in_features + self.fc.in_features
        self.out = nn.Linear(num_ftrs, outputs)

    def forward(self, x):
        # print(x.shape)  # torch.Size([16, 3, 64, 64])
        # 提取 SqueezeNet 的特征 颜色特征
        features_squeezenet = self.squeezenet.features(x)
        # print(features_squeezenet.shape)  # torch.Size([16, 512, 3, 3])

        # 将特征扩展到与原图像相同的形状
        # features_squeezenet = torch.unsqueeze(features_squeezenet, dim=2)  # .view
        # features_squeezenet = features_squeezenet.view(16, 1, 64, 64)

        # 插值上采样
        features_squeezenet_reshape = F.interpolate(features_squeezenet, size=(64, 64), mode='bilinear', align_corners=False)
        features_squeezenet_reshape = self.conv(features_squeezenet_reshape)
        # print(features_squeezenet_reshape.shape)
        # 全连接
        features_squeezenet_reshape = self.flatten(features_squeezenet_reshape)
        features_squeezenet_reshape = torch.sigmoid(self.fc(features_squeezenet_reshape))
        features_squeezenet_reshape = self.dropout(features_squeezenet_reshape)
        # print(features_squeezenet_reshape.shape)

        # 输入 ResNet
        x = self.resnet(x)
        # print(x.shape)  # torch.Size([16, 1000])

        # 将 SqueezeNet 的颜色分布预估计输出与原图像本身色彩特征输出合并 16,nums
        combined_features = torch.cat((x, features_squeezenet_reshape), dim=1)
        # print(combined_features.shape)
        out = self.out(combined_features)  # 16 1536 -- 1536 3
        out = F.normalize(out, dim=1)
        return out

    def set_param_requires_grad(self, model, feature_extract):
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False  # 冻结


model = My_CCD_NN(3)
# print(model)
