'''
[description]
对特征图进行金字塔池化操作。然后将最终特征整合到 256 维
'''
#ref:https://github.com/charlesCXK/PyTorch_Semantic_Segmentation/tree/master/DeepLabV3_PyTorch
# from torchsummary import summary
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import networks.ResNet101

class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rates=(12, 24, 36),
                 hidden_channels=256,
                 norm_act=nn.BatchNorm2d,
                 pooling_size=None):
        super(ASPP, self).__init__()
        self.pooling_size = pooling_size

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[0],
                      padding=dilation_rates[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[1],
                      padding=dilation_rates[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[2],
                      padding=dilation_rates[2])
        ])
        self.map_bn = norm_act(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)

        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.leak_relu(out)       # add activation layer
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)

        pool = self.leak_relu(pool)  # add activation layer

        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        out = self.leak_relu(out)  # add activation layer
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = (min((self.pooling_size, 0), x.shape[2]),
                            min((self.pooling_size, 1), x.shape[3]))
            padding = (
                (pooling_size[1] - 1) // 2,
                (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1,
                (pooling_size[0] - 1) // 2,
                (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1
            )

            pool = nn.functional.avg_pool2d(x, pooling_size, stride=1)
            pool = nn.functional.pad(pool, pad=padding, mode="replicate")
        return pool


class DeepLabV3(nn.Module):
    def __init__(self, class_num, bn_momentum=0.01):
        super(DeepLabV3, self).__init__()
        self.Resnet101 = ResNet101.get_resnet101(dilation=[1, 1, 1, 2], bn_momentum=bn_momentum, is_fpn=False)
        self.ASPP = ASPP(2048, 256, [6, 12, 18], norm_act=nn.BatchNorm2d)
        self.classify = nn.Conv2d(256, class_num, 1, bias=True)
        self.global_layers = nn.Sequential(
            nn.Linear(2048, 256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.dropout = nn.Dropout(p=0.5)
        self.avg_pool = nn.AvgPool2d(kernel_size=14, stride=None, padding=0)


    def forward(self, input):
        x = self.Resnet101(input) #20,2048,14,14

        aspp = self.ASPP(x)     # 空间金字塔池化    20,256,14,14
        predict = self.classify(aspp)    #20,2,14,14

        output= F.upsample(predict, size=input.size()[2:4], mode='bilinear', align_corners=True) #20,2,224,224

        x = self.avg_pool(x)#20,2048,1,1
        lv_dis=self.global_layers(self.dropout(x.view(-1, 2048)))#20,1
        return output,lv_dis