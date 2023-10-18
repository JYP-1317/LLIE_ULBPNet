import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional
import cv2
import numpy as np


class LBP(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, padding):
        super(LBP, self).__init__()
        self.asb = ASB(in_channels=input_size, out_channels=output_size, kernel_size=kernel_size, padding=padding)
        self.anb = ANB(in_channels=output_size)

    def forward(self, x):
        darken_feat = self.asb(x)
        brighten_feat = self.anb(darken_feat)

        return brighten_feat


class MASEblock(nn.Module):
    def __init__(self, in_channels, r=1):
        super().__init__()
        self.squeeze = nn.AdaptiveMaxPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)

        return x


class MISEblock(nn.Module):
    def __init__(self, in_channels, r=1):
        super().__init__()
        self.squeeze = nn.AdaptiveMaxPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = -self.squeeze(-x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)

        return x


class ANB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.maseblock = MASEblock(in_channels)
        self.miseblock = MISEblock(in_channels)

    def forward(self, x):

        im_h = self.maseblock(x)
        im_l = self.miseblock(x)

        me = torch.tensor(0.00001, dtype=torch.float32).cuda()

        x = (x - im_l) / torch.maximum(im_h - im_l, me)
        x = torch.clip(x, 0.0, 1.0)

        return x


class ASB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()

        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class DLN(nn.Module):
    def __init__(self, input_dim=3, dim=64):
        super(DLN, self).__init__()
        inNet_dim = input_dim + 3
        # 1:brightness
        self.color_in = LBP(3, 3, 3, 1)
        self.Color_map = ConvBlock(3, dim, 3, 1, 1)
        self.feature = ConvBlock(input_size=dim, output_size=dim, kernel_size=3, stride=1, padding=1)
        self.out = nn.Conv2d(in_channels=dim, out_channels=3, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_ori, tar=None):
        # feature extraction
        #x_ori = x_ori.type('torch.cuda.HalfTensor')

        Color_map_in = self.color_in(x_ori)
        Color_map_out = self.Color_map(Color_map_in)
        feature_out = self.feature(Color_map_out)
        pred = self.out(feature_out)

        return pred


############################################################################################
# Base models
############################################################################################

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True, isuseBN=False):
        super(ConvBlock, self).__init__()
        self.isuseBN = isuseBN
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        if self.isuseBN:
            self.bn = nn.BatchNorm2d(output_size)
        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.isuseBN:
            out = self.bn(out)
        out = self.act(out)
        return out


class SERESLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SERESLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        return y


class MSERESLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(MSERESLayer, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        return y


class ResBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.act1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.act2 = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        return out


class MultiBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        super(MultiBlock, self).__init__()
        self.lev1 = nn.Sequential(
            nn.Conv2d(input_size, input_size, 3, 1, 1),
            nn.ReLU()
        )
        self.lev2 = nn.Sequential(
            nn.Conv2d(input_size, input_size, 3, 2, 1),
            nn.Conv2d(input_size, input_size, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(input_size, input_size, 2, 2)
        )
        self.lev3 = nn.Sequential(
            nn.Conv2d(input_size, input_size, 3, 4, 1),
            nn.Conv2d(input_size, input_size, 1, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(input_size, input_size, 4, 4)
        )
        self.oconv = nn.Sequential(
            nn.Conv2d(4 * input_size, output_size, 1, 1, 0),
            nn.ReLU()
        )

    def forward(self, x):
        l1 = self.lev1(x)
        l2 = self.lev2(x)
        l3 = self.lev3(x)
        total_lev = torch.cat([x, l1, l2, l3], dim=1)
        out = self.oconv(total_lev)
        return out