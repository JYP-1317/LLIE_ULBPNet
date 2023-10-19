from pickletools import uint8

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.transforms.functional


class LightenBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(LightenBlock, self).__init__()
        self.conv_Encoder = ASB(input_size, output_size, kernel_size=3)
        self.conv_Offset = ASB(input_size, output_size, kernel_size=3)
        self.conv_Decoder = ASB(input_size, output_size, kernel_size=3)

    def forward(self, x):
        offset = self.conv_Offset(x)
        code_lighten = x + offset
        out = self.conv_Decoder(code_lighten)
        return out


class DarkenBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DarkenBlock, self).__init__()
        self.conv_Encoder = ASB(input_size, output_size, kernel_size=3)
        self.conv_Offset = ASB(input_size, output_size, kernel_size=3)
        self.conv_Decoder = ASB(input_size, output_size, kernel_size=3)

    def forward(self, x):
        offset = self.conv_Offset(x)
        code_lighten = x - offset
        out = self.conv_Decoder(code_lighten)
        return out


class FusionLayer(nn.Module):
    def __init__(self, inchannel, outchannel, reduction=16):
        super(FusionLayer, self).__init__()
        self.MASE = MASEblock(inchannel)
        self.PASB = ASB(inchannel, outchannel, kernel_size=3)

    def forward(self, x):
        b, c, _, _ = x.size()
        # y = self.avg_pool(x).view(b, c)
        y = self.MASE(x).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        y = y + x
        y = self.PASB(y)
        return y


class LBP(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(LBP, self).__init__()
        self.fusion = FusionLayer(input_size, output_size)
        self.conv1 = LightenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = DarkenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = LightenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1_1 = weASB(input_size, output_size, kernel_size=1)
        self.local_weight2_1 = weASB(input_size, output_size, kernel_size=1)

    def forward(self, x):
        x = self.fusion(x)
        hr = self.conv1(x)
        lr = self.conv2(hr)
        residue = self.local_weight1_1(x) - lr
        h_residue = self.conv3(residue)
        hr_weight = self.local_weight2_1(hr)
        return hr_weight + h_residue


class DLN(nn.Module):
    def __init__(self, input_dim=3, dim=64):
        super(DLN, self).__init__()
        inNet_dim = input_dim + 3
        # Stage 1

        self.conv_o = torch.nn.Conv2d(input_dim, 3, kernel_size=3, stride=1, padding='same')
        self.conv_s = torch.nn.Conv2d(1, 3, kernel_size=3, stride=1, padding='same')
        self.conv_h = torch.nn.Conv2d(input_dim, 3, kernel_size=3, stride=1, padding='same')

        self.se_os = SELayer(channel=3, reduction=1)
        self.se_sh = SELayer(channel=3, reduction=1)

        self.conv_wos = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding='same')
        self.conv_wsh = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding='same')

        self.enc1_1 = ConvBlock(input_size=inNet_dim, output_size=64, kernel_size=3, stride=1, padding='same')
        self.enc1_2 = ConvBlock(input_size=64, output_size=64, kernel_size=3, stride=1, padding='same')

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = ConvBlock(input_size=64, output_size=128, kernel_size=3, stride=1, padding='same')
        self.enc2_2 = ConvBlock(input_size=128, output_size=128, kernel_size=3, stride=1, padding='same')

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = ConvBlock(input_size=128, output_size=256, kernel_size=3, stride=1, padding='same')
        self.enc3_2 = ConvBlock(input_size=256, output_size=256, kernel_size=3, stride=1, padding='same')

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # Expansive path
        self.unpool3 = upsample(input_size=256, output_size=128,
                                kernel_size=2, stride=2, padding=0)

        self.dec4_2 = ConvBlock(input_size=2 * 128, output_size=256, kernel_size=3, stride=1, padding='same')
        self.dec4_1 = ConvBlock(input_size=256, output_size=128, kernel_size=3, stride=1, padding='same')

        self.unpool4 = upsample(input_size=128, output_size=64,
                                kernel_size=2, stride=2, padding=0)

        self.dec5_2 = ConvBlock(input_size=2 * 64, output_size=64, kernel_size=3, stride=1, padding='same')
        self.dec5_1 = ConvBlock(input_size=64, output_size=64, kernel_size=3, stride=1, padding='same')

        self.lbp = LBP(input_size=256, output_size=256, kernel_size=3, stride=1, padding='same')

        self.ms = MSBlock(input_size=128, output_size=128, kernel_size=3)

        self.no = NoiseBlock(input_size=64, output_size=64, kernel_size=3)

        self.out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)

        # STAGE 2
        self.se_os2 = SELayer(channel=3, reduction=1)
        self.se_sh2 = SELayer(channel=3, reduction=1)

        self.conv_wos2 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding='same')
        self.conv_wsh2 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding='same')

        self.enc2_1_1 = ConvBlock(input_size=inNet_dim, output_size=64, kernel_size=3, stride=1, padding='same')
        self.enc2_1_2 = ConvBlock(input_size=64, output_size=64, kernel_size=3, stride=1, padding='same')

        self.pool2_1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_2_1 = ConvBlock(input_size=64, output_size=128, kernel_size=3, stride=1, padding='same')
        self.enc2_2_2 = ConvBlock(input_size=128, output_size=128, kernel_size=3, stride=1, padding='same')

        self.pool2_2 = nn.MaxPool2d(kernel_size=2)

        self.enc2_3_1 = ConvBlock(input_size=128, output_size=256, kernel_size=3, stride=1, padding='same')
        self.enc2_3_2 = ConvBlock(input_size=256, output_size=256, kernel_size=3, stride=1, padding='same')

        self.pool2_3 = nn.MaxPool2d(kernel_size=2)

        # Expansive path
        self.unpool2_3 = upsample(input_size=256, output_size=128,
                                  kernel_size=2, stride=2, padding=0)

        self.dec2_4_2 = ConvBlock(input_size=2 * 128, output_size=256, kernel_size=3, stride=1, padding='same')
        self.dec2_4_1 = ConvBlock(input_size=256, output_size=128, kernel_size=3, stride=1, padding='same')

        self.unpool2_4 = upsample(input_size=128, output_size=64,
                                  kernel_size=2, stride=2, padding=0)

        self.dec2_5_2 = ConvBlock(input_size=2 * 64, output_size=64, kernel_size=3, stride=1, padding='same')
        self.dec2_5_1 = ConvBlock(input_size=64, output_size=64, kernel_size=3, stride=1, padding='same')

        self.lbp2 = LBP(input_size=256, output_size=256, kernel_size=3, stride=1, padding='same')

        self.ms2 = MSBlock(input_size=128, output_size=128, kernel_size=3)

        self.no2 = NoiseBlock(input_size=64, output_size=64, kernel_size=3)

        self.asab = ASAN(3, 3, 3)
        self.asabout = ConvBlock(3, 64, 3, 1, 1)

        self.featout = ConvBlock(input_size=dim, output_size=dim, kernel_size=3, stride=1, padding=1)

        self.out2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1)  # 수정

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

    def rgb2hsv(self, rgb: torch.Tensor) -> torch.Tensor:
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        cmin = torch.min(rgb, dim=1, keepdim=True)[0]
        delta = cmax - cmin
        hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)

        return hsv_s

    def forward(self, x_ori, tar=None):
        # data gate

        x_s = self.rgb2hsv(x_ori)
        x_s = 1 - x_s

        xh = (255 * x_ori).type('torch.cuda.ByteTensor')
        x_h = torchvision.transforms.functional.equalize(xh)
        x_h = (x_h / 255).type('torch.cuda.FloatTensor')


        x_oric = self.conv_o(x_ori)
        x_sc = self.conv_s(x_s)
        x_hc = self.conv_h(x_h)

        # STAGE 1
        weight_os = x_oric * x_sc
        weight_sh = x_sc * x_hc

        weight_os = self.se_os(weight_os)
        weight_sh = self.se_sh(weight_sh)

        weight_os = self.conv_wos(weight_os)
        weight_sh = self.conv_wsh(weight_sh)

        os_feature = x_ori + weight_os
        sh_feature = x_h + weight_sh

        x_in = torch.cat((os_feature, sh_feature), 1)

        feature1 = self.enc1_1(x_in)
        feature1 = self.enc1_2(feature1)
        pool_feature1 = self.pool1(feature1)

        feature2 = self.enc2_1(pool_feature1)
        feature2 = self.enc2_2(feature2)
        pool_feature2 = self.pool1(feature2)

        feature3 = self.enc3_1(pool_feature2)
        feature3 = self.enc3_2(feature3)

        LBP_feature3 = self.lbp(feature3)

        feature4 = self.unpool3(LBP_feature3)
        feature4 = torch.cat((feature2, feature4), 1)
        feature4 = self.dec4_2(feature4)
        feature4 = self.dec4_1(feature4)

        MS_feature4 = self.ms(feature4)

        feature5 = self.unpool4(MS_feature4)
        feature5 = torch.cat((feature1, feature5), 1)
        feature5 = self.dec5_2(feature5)
        feature5 = self.dec5_1(feature5)

        NO_feature6 = self.no(feature5)
        Stage_1_out = self.out(NO_feature6)

        # STAGE 2

        weight_os2 = x_oric * x_sc
        weight_sh2 = x_sc * Stage_1_out

        weight_os2 = self.se_os2(weight_os2)
        weight_sh2 = self.se_sh2(weight_sh2)

        weight_os2 = self.conv_wos2(weight_os2)
        weight_sh2 = self.conv_wsh2(weight_sh2)

        os_feature2 = x_ori + weight_os2
        sh_feature2 = Stage_1_out + weight_sh2

        x_in2 = torch.cat((os_feature2, sh_feature2), 1)

        feature2_1 = self.enc2_1_1(x_in2)
        feature2_1 = self.enc2_1_2(feature2_1)
        pool_feature2_1 = self.pool2_1(feature2_1)

        feature2_2 = self.enc2_2_1(pool_feature2_1)
        feature2_2 = self.enc2_2_2(feature2_2)
        pool_feature2_2 = self.pool2_1(feature2_2)

        feature2_3 = self.enc2_3_1(pool_feature2_2)
        feature2_3 = self.enc2_3_2(feature2_3)

        LBP_feature2_3 = self.lbp2(feature2_3)

        feature2_4 = self.unpool2_3(LBP_feature2_3)
        feature2_4 = torch.cat((feature2_2, feature2_4), 1)
        feature2_4 = self.dec2_4_2(feature2_4)
        feature2_4 = self.dec2_4_1(feature2_4)

        MS_feature2_4 = self.ms2(feature2_4)

        feature2_5 = self.unpool2_4(MS_feature2_4)
        feature2_5 = torch.cat((feature2_1, feature2_5), 1)
        feature2_5 = self.dec2_5_2(feature2_5)
        feature2_5 = self.dec2_5_1(feature2_5)

        NO_feature2_6 = self.no2(feature2_5)

        asbanb_in = self.asab(x_ori)
        asbanb_out = self.asabout(asbanb_in)

        final_out = NO_feature2_6 * asbanb_out
        colorbright = self.featout(final_out)
        pred = self.out2(colorbright)

        return pred

###########################################################################################################
# scaling part
###########################################################################################################

class MSBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size):
        super(MSBlock, self).__init__()
        self.ds1 = nn.Conv2d(input_size, output_size, kernel_size, stride=2, padding=1)
        self.ds2 = nn.Conv2d(input_size, output_size, kernel_size, stride=2, padding=1)
        self.conv1 = nn.Conv2d(input_size, output_size, kernel_size, stride=1, padding='same', bias=True)
        self.conv2 = nn.Conv2d(input_size, output_size, kernel_size, stride=1, padding='same', bias=True)
        self.conv3 = nn.Conv2d(input_size, output_size, kernel_size, stride=1, padding='same', bias=True)
        self.conv4 = nn.Conv2d(input_size * 4, output_size, kernel_size, stride=1, padding='same', bias=True)
        self.bn1 = nn.BatchNorm2d(input_size)
        self.bn2 = nn.BatchNorm2d(input_size)
        self.bn3 = nn.BatchNorm2d(input_size)
        self.act1 = nn.ReLU()

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.act1(out1)  # CONV BLOCK

        out2 = self.ds1(x)
        out2 = self.conv2(out2)
        out2 = self.bn2(out2)
        out2 = self.act1(out2)
        out2 = F.interpolate(out2, size=(out1.size()[2], out1.size()[3]))

        out3 = self.ds1(x)
        out3 = self.ds2(out3)
        out3 = self.conv3(out3)
        out3 = self.bn3(out3)
        out3 = self.act1(out3)
        out3 = F.interpolate(out3, size=(out2.size()[2], out2.size()[3]))
        out3 = F.interpolate(out3, size=(out1.size()[2], out1.size()[3]))
        out = torch.cat([x, out1, out2, out3], dim=1)
        out = self.conv4(out)

        return out


class NoiseBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size):
        super(NoiseBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_size, output_size, kernel_size, stride=1, padding='same', bias=True)
        self.conv2 = nn.Conv2d(input_size, output_size, kernel_size, stride=1, padding='same', bias=True)
        self.conv3 = nn.Conv2d(input_size, output_size, kernel_size, stride=1, padding='same', bias=True)
        self.act1 = nn.ReLU()

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.act1(out1)
        out2 = self.conv2(out1)
        out2 = self.act1(out2)
        out3 = self.conv3(out2)

        out = torch.tanh(out3)

        return out


###########################################################################################################
# Stretching part
###########################################################################################################

class MASEblock(nn.Module):
    def __init__(self, in_channels, r=16):
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
    def __init__(self, in_channels, r=16):
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
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class weASB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        x = self.conv(x)

        return x


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
        self.act = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.isuseBN:
            out = self.bn(out)
        out = self.act(out)
        return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DeconvBlock, self).__init__()

        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.deconv(x)

        return self.act(out)


class UpBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(UpBlock, self).__init__()

        self.conv1 = DeconvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = ConvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        hr = self.conv1(x)
        lr = self.conv2(hr)
        residue = self.local_weight1(x) - lr
        h_residue = self.conv3(residue)
        hr_weight = self.local_weight2(hr)
        return hr_weight + h_residue


class DownBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(DownBlock, self).__init__()

        self.conv1 = ConvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = ConvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        lr = self.conv1(x)
        hr = self.conv2(lr)
        residue = self.local_weight1(x) - hr
        l_residue = self.conv3(residue)
        lr_weight = self.local_weight2(lr)
        return lr_weight + l_residue


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(num_filter)

        self.act1 = torch.nn.PReLU()
        self.act2 = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = self.act2(out)

        return out


class upsample(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding='same'):
        super(upsample, self).__init__()
        self.up = nn.ConvTranspose2d(input_size, output_size, kernel_size=2, stride=2, padding=0, bias=True)

    def forward(self, x):
        x = self.up(x)

        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
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
        return x * y.expand_as(x)

############################################################################################
# origin ASBANB
############################################################################################
class ASAN(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size):
        super(ASAN, self).__init__()
        self.asb = siASB(in_channels=input_size, out_channels=output_size, kernel_size=kernel_size)
        self.anb = siANB(in_channels=output_size)

    def forward(self, x):
        darken_feat = self.asb(x)
        brighten_feat = self.anb(darken_feat)

        return brighten_feat


class siMASEblock(nn.Module):
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


class siMISEblock(nn.Module):
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


class siANB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.maseblock = siMASEblock(in_channels)
        self.miseblock = siMISEblock(in_channels)

    def forward(self, x):

        im_h = self.maseblock(x)
        im_l = self.miseblock(x)

        me = torch.tensor(0.00001, dtype=torch.float32).cuda()

        x = (x - im_l) / torch.maximum(im_h - im_l, me)
        x = torch.clip(x, 0.0, 1.0)

        return x


class siASB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)

        return x

