import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

def Double_Conv2d(in_channels, out_channels, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding),
        nn.BatchNorm2d(out_channels),
        # nn.ReLU(inplace=True),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=padding),
        nn.BatchNorm2d(out_channels),
        # nn.ReLU(inplace=True),
        nn.LeakyReLU(inplace=True),
    )

def DeConv2D(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_channels),
        # nn.ReLU(inplace=True),
        nn.LeakyReLU(inplace=True),
    )

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(384, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.num_filters = 64
        self.doubleb = Double_Conv2d(self.num_filters*32, self.num_filters*4, padding=1)
        self.up4 = DeConv2D(self.num_filters*4, self.num_filters*4)
        self.up3 = DeConv2D(self.num_filters*8, self.num_filters*4)
        self.up2 = DeConv2D(self.num_filters*4, self.num_filters*2)
        self.up1 = DeConv2D(self.num_filters*2, self.num_filters)

        self.double4r = Double_Conv2d(self.num_filters*20, self.num_filters*8, padding=1)
        self.double3r = Double_Conv2d(self.num_filters*12, self.num_filters*4, padding=1)
        self.double2r = Double_Conv2d(self.num_filters*6, self.num_filters*2, padding=1)
        self.double1r = Double_Conv2d(self.num_filters*3, self.num_filters, padding=1)
        
        self.num_class = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.sig = nn.Sigmoid()
        # self.sig = nn.Softmax()
        # self._init_weight()

    def forward(self, d0, d1, d2, d3, d4):
        # x = self.up4(d4)
        x = torch.cat([d4, d3], dim=1)
        x = self.double4r(x)

        x = self.up3(x)
        x = torch.cat([x, d2], dim=1)
        x = self.double3r(x)

        x = self.up2(x)
        x = torch.cat([x, d1], dim=1)
        x = self.double2r(x)

        # x = self.up1(x)
        x = torch.cat([x, d0], dim=1)
        x = self.double1r(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder_deeplabv3(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder_deeplabv3, self).__init__()
        if backbone == 'resnet101' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Decoder_noAspp(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder_noAspp, self).__init__()
        self.conv1 = nn.Conv2d(384, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.num_filters = 64
        self.doubleb = Double_Conv2d(self.num_filters*32, self.num_filters*4, padding=1)
        self.up4 = DeConv2D(self.num_filters*32, self.num_filters*4)
        self.up3 = DeConv2D(self.num_filters*8, self.num_filters*4)
        self.up2 = DeConv2D(self.num_filters*4, self.num_filters*2)
        self.up1 = DeConv2D(self.num_filters*2, self.num_filters)

        self.double4r = Double_Conv2d(self.num_filters*20, self.num_filters*8, padding=1)
        self.double3r = Double_Conv2d(self.num_filters*12, self.num_filters*4, padding=1)
        self.double2r = Double_Conv2d(self.num_filters*6, self.num_filters*2, padding=1)
        self.double1r = Double_Conv2d(self.num_filters*3, self.num_filters, padding=1)
        
        self.num_class = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.sig = nn.Sigmoid()
        # self.sig = nn.Softmax()
        # self._init_weight()

    def forward(self, d0, d1, d2, d3, d4):
        d4 = self.maxpool(d4)
        d4 = self.up4(d4)
        x = torch.cat([d4, d3], dim=1)        
        x = self.double4r(x)

        x = self.up3(x)
        x = torch.cat([x, d2], dim=1)
        x = self.double3r(x)

        x = self.up2(x)
        x = torch.cat([x, d1], dim=1)
        x = self.double2r(x)

        # x = self.up1(x)
        x = torch.cat([x, d0], dim=1)
        x = self.double1r(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)

def build_deeplabv3_decoder(num_classes, backbone, BatchNorm):
    return Decoder_deeplabv3(num_classes, backbone, BatchNorm)

def build_noAspp_decoder(num_classes, backbone, BatchNorm):
    return Decoder_noAspp(num_classes, backbone, BatchNorm)