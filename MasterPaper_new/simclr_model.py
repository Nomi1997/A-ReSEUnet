# import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

RETURN_PREACTIVATION = False  # return features from the model, if false return classification logits
NUM_CLASSES = 2  # only used if RETURN_PREACTIVATION = False

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, mask = None):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SimCLR_addDecoder(nn.Module):
    def __init__(self, freeze=False, num_classes=1):
        super(SimCLR_addDecoder, self).__init__()
        self.freeze = freeze
        self.inplanes = 64
        dilations = [1, 1, 1, 2]
        strides = [1, 2, 2, 2]
        BatchNorm = nn.BatchNorm2d
        block = BasicBlock
        layers = [2, 2, 2, 2]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        
        self.num_filters = 64
        self.doubleb = Double_Conv2d(self.num_filters*8, self.num_filters*16, padding=1)
        self.up1 = DeConv2D(self.num_filters*16, self.num_filters*8)
        self.up2 = DeConv2D(self.num_filters*8, self.num_filters*4)
        self.up3 = DeConv2D(self.num_filters*4, self.num_filters*2)
        self.up4 = DeConv2D(self.num_filters*2, self.num_filters)

        self.double1r = Double_Conv2d(self.num_filters*16, self.num_filters*8, padding=1)
        self.double2r = Double_Conv2d(self.num_filters*8, self.num_filters*4, padding=1)
        self.double3r = Double_Conv2d(self.num_filters*4, self.num_filters*2, padding=1)
        self.double4r = Double_Conv2d(self.num_filters*2, self.num_filters, padding=1)

        self.final = nn.Conv2d(self.num_filters, num_classes, kernel_size=1)

        self.classifier = nn.Conv2d(2048, 20, 1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 4)
        self.act = nn.Sigmoid()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        with torch.set_grad_enabled(self.freeze):
            x0 = self.conv1(x)
            x0 = self.bn1(x0)
            x0 = self.relu(x0)
            x0 = self.maxpool(x0)

            x1 = self.layer1(x0)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            x = self.maxpool(x4)

            x5 = self.doubleb(x)

        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.double1r(x)

        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.double2r(x)

        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)                                                                              
        x = self.double3r(x)

        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.double4r(x)

        x = self.final(x)

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        # x = self.act(x)
        return x

def Double_Conv2d(in_channels, out_channels, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

def DeConv2D(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

def cropping(x, y):
    if x.shape == y.shape:
        return x
    else:
        return x[:, :, :y.shape[2], :y.shape[3]]

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()

        self.num_filters = 64
        
        self.double1l = Double_Conv2d(in_channels, self.num_filters, padding=1)
        self.double2l = Double_Conv2d(self.num_filters, self.num_filters*2, padding=1)
        self.double3l = Double_Conv2d(self.num_filters*2, self.num_filters*4, padding=1)
        self.double4l = Double_Conv2d(self.num_filters*4, self.num_filters*8, padding=1)
        self.doubleb = Double_Conv2d(self.num_filters*8, self.num_filters*16, padding=1)
        
        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpooling1 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.num_filters, kernel_size=2, stride=2)
        self.maxpooling2 = nn.Conv2d(in_channels=self.num_filters*2, out_channels=self.num_filters*2, kernel_size=2, stride=2)
        self.maxpooling3 = nn.Conv2d(in_channels=self.num_filters*4, out_channels=self.num_filters*4, kernel_size=2, stride=2)
        self.maxpooling4 = nn.Conv2d(in_channels=self.num_filters*8, out_channels=self.num_filters*8, kernel_size=2, stride=2)

        self.up1 = DeConv2D(self.num_filters*16, self.num_filters*8)
        self.up2 = DeConv2D(self.num_filters*8, self.num_filters*4)
        self.up3 = DeConv2D(self.num_filters*4, self.num_filters*2)
        self.up4 = DeConv2D(self.num_filters*2, self.num_filters)

        self.double1r = Double_Conv2d(self.num_filters*16, self.num_filters*8, padding=1)
        self.double2r = Double_Conv2d(self.num_filters*8, self.num_filters*4, padding=1)
        self.double3r = Double_Conv2d(self.num_filters*4, self.num_filters*2, padding=1)
        self.double4r = Double_Conv2d(self.num_filters*2, self.num_filters, padding=1)

        self.final = nn.Conv2d(self.num_filters, out_channels, kernel_size=1)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        l1 = self.double1l(x)
        x = self.maxpooling(l1)
        # x = self.maxpooling1(l1)

        l2 = self.double2l(x)
        x = self.maxpooling(l2)
        # x = self.maxpooling2(l2)

        l3 = self.double3l(x)
        x = self.maxpooling(l3)
        # x = self.maxpooling3(l3)

        l4 = self.double4l(x)
        x = self.maxpooling(l4)
        # x = self.maxpooling4(l4)

        x = self.doubleb(x)

        x = self.up1(x)
        l4 = cropping(l4, x)
        x = torch.cat([l4, x], dim=1)
        x = self.double1r(x)

        x = self.up2(x)
        l3 = cropping(l3, x)
        x = torch.cat([l3, x], dim=1)
        x = self.double2r(x)

        x = self.up3(x)
        l2 = cropping(l2, x)
        x = torch.cat([l2, x], dim=1)
        x = self.double3r(x)

        x = self.up4(x)
        l1 = cropping(l1, x)
        x = torch.cat([l1, x], dim=1)
        x = self.double4r(x)

        x = self.final(x)
        # x = self.act(x)
        return x
