#!/usr/bin/env python3.6

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.nn import init

class unetConv1(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=1, ks=3, stride=1, padding=1):
        super(unetConv1, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size


    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x

class unetUp3(nn.Module):
    def __init__(self, in_size, out_size, n_concat=2):
        super(unetUp3, self).__init__()
        self.conv = unetConv1(in_size*2 + (n_concat - 2) * out_size, out_size, False)
    def forward(self, high_feature, *low_feature):
        for feature in low_feature:
            outputs0 = torch.cat([high_feature, feature], 1)
        return self.conv(outputs0)

class unetUp2(nn.Module):
    def __init__(self, in_size, out_size, n_concat=2):
        super(unetUp2, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=2, padding=0)
        self.conv = unetConv1(in_size + (n_concat - 2) * out_size, out_size, False)
    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0,self.up( feature)], 1)
        return self.conv(outputs0)

class unetUp1(nn.Module):
    def __init__(self, in_size, out_size, n_concat=2):
        super(unetUp1, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=2, padding=0)
        self.conv = unetConv1(int(out_size*2) + (n_concat - 2) * out_size, out_size, False)

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0,self.up( feature)], 1)
        return self.conv(outputs0)

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, n_concat=2):
        super(unetUp, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=12, stride=4, padding=2)
        self.conv = unetConv1(int(out_size*2) + (n_concat - 2) * out_size, out_size, False)

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0,self.up( feature)], 1)
        return self.conv(outputs0)



class Alex_Unet(nn.Module):
    def __init__(self,n_classifi=1,n_classes_de=2):
        super().__init__()
        self.alexnet = models.alexnet(pretrained=False).features
        self.alexnet.add_module('0',nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)))
        self.fc = nn.Linear(256, n_classifi)
        self.avg_pool = nn.AvgPool2d(kernel_size=6, stride=None, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        #decoder
        self.decoder = Decoder(n_classes_de)

    @property
    def features(self):
        return self.alexnet

    @property
    def classifier(self):
        return self.fc

    def forward(self, x):
        feature = self.features(x)
        features=[]


        for index,layer in enumerate(self.alexnet):
            x = layer(x) # 64,55,55   64,27,27  192,13,13  384,13,13  256,6,6
            if (index in [0,3,6,8,10]):
                features.append(x)

        out = self.avg_pool(feature)  # num,256,1,1

        lv_dis = self.classifier(self.dropout(out.view(-1,256)))
        heat = self.decoder(features)
        return heat,lv_dis

class Decoder(nn.Module):
    def __init__(self,n_classes):
        super(Decoder, self).__init__()
        filters=[64,192,384,256]
        self.up_concat3 = unetUp3(filters[3], filters[2])#num,384,19,19
        self.up_concat2 = unetUp2(filters[2], filters[1])
        self.up_concat1 = unetUp1(filters[1] , filters[0] )
        self.up_concat = unetUp(filters[0], n_classes)

    def forward(self, features):#num,256,9,9

        up3 = self.up_concat3(features[4], features[3]) #num,384,19,19
        up2 = self.up_concat2(up3, features[2]) #num,192,39,39
        up1 = self.up_concat1(up2, features[1])#num,64,79,79
        up = self.up_concat(up1, features[0]) #num x,256,256

        return up


#ref https://blog.csdn.net/weixin_46040552/article/details/103735028
# class Decoder(nn.Module):
#     def __init__(self, num_classes, filter_multiplier ):
#         super(Decoder, self).__init__()
#         low_level_inplanes = filter_multiplier
#         C_low = 48
#         # 构建卷积和BN层。
#         self.conv1 = nn.Conv2d(low_level_inplanes, C_low, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(48)
#         # 构建最后的卷积操作层。
#         self.last_conv = nn.Sequential(nn.Conv2d(304,256, kernel_size=3, stride=1, padding=1, bias=False),
#                                        nn.BatchNorm2d(256),
#                                        nn.Dropout(0.5),
#                                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
#                                        nn.BatchNorm2d(256),
#                                        nn.Dropout(0.1),
#                                        nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
#         # initialise weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init_weights(m, init_type='kaiming')
#             elif isinstance(m, nn.BatchNorm2d):
#                 init_weights(m, init_type='kaiming')
#
#     def forward(self, x, low_level_feat):
#         # 对低层网络层的输出进行卷积和BN操作，使维度相符。
#         low_level_feat = self.conv1(low_level_feat)
#         low_level_feat = self.bn1(low_level_feat)
#         # 进行插值操作。
#         x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
#         # 将低级和高级的特征组合到一起。
#         x = torch.cat((x, low_level_feat), dim=1)
#         # 进行最后的一系列卷积操作。
#         x = self.last_conv(x)
#         return x