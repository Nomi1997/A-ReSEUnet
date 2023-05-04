import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp, build_aspp_deepLabv3
from modeling.decoder import build_decoder, build_deeplabv3_decoder, build_noAspp_decoder
from modeling.backbone import build_backbone
from torchsummary import summary
import math
import matplotlib.pyplot as plt

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet101', output_stride=16, num_classes=1, sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp_deepLabv3(backbone, output_stride, BatchNorm)
        self.decoder = build_deeplabv3_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

class DilatedConvBlock(nn.Module):
    def __init__(self, in_c=3, out_c=32, **kwargs):
        super(DilatedConvBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_c, out_c, **kwargs),
            nn.BatchNorm2d(out_c),
            # nn.InstanceNorm2d(out_c),
            # nn.ReLU(True),
            nn.LeakyReLU(True),
        )
    def forward(self, x):
        # a = self.main.parameters()
        # for param in self.main.parameters():
        #     b = param.data
        # c = self.main(x)

        # plt.imshow(c[0][0].detach().numpy())
        # plt.show()

        return self.main(x)

class DilatedConv2ResBlock(nn.Module):
    """
    Residual convolutional block with dilated filters. 
    """
    def __init__(self, in_c=3, out_c=32, **kwargs):
        super(DilatedConv2ResBlock, self).__init__()

        self.in_c   = in_c
        self.out_c  = out_c

        # Residual connection
        self.res    = nn.Sequential(
            nn.Conv2d(in_c, out_c, **kwargs),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
            nn.Conv2d(out_c, out_c, **kwargs),
            nn.BatchNorm2d(out_c),
        )

        if out_c != in_c:
            # Mapping connection. 
            self.mapper = nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_c),
            )
        # self.relu   = nn.ReLU(True)
        self.relu   = nn.LeakyReLU(True)

    def forward(self, x, **kwargs):
        residual        = self.res(x)
        if self.in_c != self.out_c:
            x           = self.mapper(x)

        out             = self.relu(x + residual)
        # plt.imshow(out[0][0].detach().numpy())
        # plt.show()
        return out


class Dilated10ConvAttentionMap1x1AvgTauWithSparsity(nn.Module):
    def __init__(self, sparse_radio):
        super(Dilated10ConvAttentionMap1x1AvgTauWithSparsity, self).__init__()

        self.nc = 3
        self.ndf = 64

        self.attention_net  = nn.Sequential(
            DilatedConvBlock(self.nc, self.ndf, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            DilatedConv2ResBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            DilatedConv2ResBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            DilatedConv2ResBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=3, dilation=3, bias=False),
            DilatedConv2ResBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=5, dilation=5, bias=False),
            DilatedConv2ResBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=10, dilation=10, bias=False),
            DilatedConv2ResBlock(self.ndf, self.ndf, kernel_size=3, stride=1, padding=20, dilation=20, bias=False),
            nn.Conv2d(self.ndf, 1, kernel_size=1, stride=1, padding=0),
        )

        # self.first records whether the first iteration is over or not. If it is, then the inferred tau
        #   from the training batch is used to set self.tau.
        # Otherwise, self.tau = 0.9 * self.tau + 0.1 * self.tau_new is used. 
        self.register_buffer('first', torch.ByteTensor([1]))
        # The learnt threshold for the compressed sigmoid. 
        self.register_buffer('tau', torch.FloatTensor([0.]))
        # Required sparsity 所需的稀疏性 -> 比較重要!!!
        self.p              = sparse_radio
        # Compression of the sigmoid. sigmoid壓縮
        self.r              = 20
        # The compressed and biased sigmoid
        self.sigmoid        = lambda x: torch.sigmoid(self.r * (x))
        self._init_weight()

    def forward(self, images, train=True):

        # Get the confidence map. 
        att_small = self.attention_net(images)
        if train:
            # In train phase, use the tau obtained from this training batch to 
            # compute the threshold. 
            A = att_small.detach().view(att_small.size(0), -1).contiguous()
            A, _ = torch.sort(A, dim=1, descending=True)
            t_idx = int(np.floor(self.p * A.size(1)))
            tau = torch.mean(A[:, t_idx]).item()
            # If no training batches have been seen so far. 
            if self.first.item():
                self.tau.fill_(tau)
                self.first.fill_(0)
            # Else, use the following formula to update self.tau
            else:
                self.tau = 0.9 * self.tau + 0.1 * tau
        else:
            # In the testing phase, use the learnt tau.
            tau = self.tau

        # Activate the confidence maps using the sigmoid.
        # plt.imshow(attention[0][0].detach().numpy())
        # plt.show()

        attention = self.sigmoid(att_small - tau)
        a =  images * ((1 - attention) * 1 + 1)        

        return a, 1 - attention

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Master_all(nn.Module):
    def __init__(self, sparse_radio=0.6, backbone='resnet50', output_stride=16, num_classes=1, sync_bn=True, freeze_bn=False):
        super(Master_all, self).__init__()
        if backbone == 'drn':
            output_stride = 8
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.project = nn.Sequential( 
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(256, 512, bias=False),
                nn.ReLU(),
                nn.Linear(512, 128, bias=False),
            )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fla = nn.Flatten()
        self.lin = nn.Linear(256, 512)
        
        self.att = Dilated10ConvAttentionMap1x1AvgTauWithSparsity(sparse_radio)
        self.sig = nn.Sigmoid()

        self.norm_conv = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True),)

        self.last_conv = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                    #    nn.ReLU(),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                    #    nn.ReLU(),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.1))
        # self.num_class = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.num_class_new = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.freeze_bn = freeze_bn

    def forward(self, input):       
        if self.training:
            tmp = input.shape[0] // 3
            z1, z2 = self.att(input[:tmp,:,:,:])
            input = torch.cat([z1, input[tmp:,:,:,:]], dim=0)

            d, d0, d1, d2, d3, x = self.backbone(input, tmp)
            x = self.aspp(x)
            x1 = x[:tmp,:,:,:]
            x2 = x[tmp:,:,:,:]
            y = self.project(x2)

            x = self.decoder(d0, d1, d2, d3, x1)

            x = F.interpolate(x, size=d.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, d], dim=1)
            x = self.norm_conv(x)
            x = self.last_conv(x)
            x = self.num_class_new(x)
            x = self.sig(x)

            x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
            return x, y, z2, z1
        else:
            z1, z2 = self.att(input)
            d, d0, d1, d2, d3, x = self.backbone(z1)
            x = self.aspp(x)
            x = self.decoder(d0, d1, d2, d3, x)

            x = F.interpolate(x, size=d.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, d], dim=1)
            x = self.norm_conv(x)
            x = self.last_conv(x)
            x = self.num_class_new(x)
            x = self.sig(x)

            x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
            return x, z2, z1

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

class Master_noAspp(nn.Module):
    def __init__(self, sparse_radio=0.6, backbone='resnet50', output_stride=16, num_classes=1, sync_bn=True, freeze_bn=False):
        super(Master_noAspp, self).__init__()
        if backbone == 'drn':
            output_stride = 8
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.decoder = build_noAspp_decoder(num_classes, backbone, BatchNorm)
        self.project = nn.Sequential( 
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(2048, 512, bias=False),
                nn.ReLU(),
                nn.Linear(512, 128, bias=False),
            )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fla = nn.Flatten()
        self.lin = nn.Linear(256, 512)
        
        self.att = Dilated10ConvAttentionMap1x1AvgTauWithSparsity(sparse_radio)
        self.sig = nn.Sigmoid()

        self.norm_conv = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True),)

        self.last_conv = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                    #    nn.ReLU(),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                    #    nn.ReLU(),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.1))
        self.num_class = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.freeze_bn = freeze_bn

    def forward(self, input):       
        if self.training:
            tmp = input.shape[0] // 3
            z1, z2 = self.att(input[:tmp,:,:,:])
            input = torch.cat([z1, input[tmp:,:,:,:]], dim=0)

            d, d0, d1, d2, d3, x = self.backbone(input, tmp)
            x1 = x[:tmp,:,:,:]
            x2 = x[tmp:,:,:,:]
            y = self.project(x2)

            x = self.decoder(d0, d1, d2, d3, x1)

            x = F.interpolate(x, size=d.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, d], dim=1)
            x = self.norm_conv(x)
            x = self.last_conv(x)
            x = self.num_class(x)
            x = self.sig(x)

            x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
            return x, y, z2, z1
        else:
            z1, z2 = self.att(input)
            d, d0, d1, d2, d3, x = self.backbone(z1)
            x = self.decoder(d0, d1, d2, d3, x)

            x = F.interpolate(x, size=d.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, d], dim=1)
            x = self.norm_conv(x)
            x = self.last_conv(x)
            x = self.num_class(x)
            x = self.sig(x)

            x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
            return x, z2, z1

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()



class Master_noAtt(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=5,
                 sync_bn=True, freeze_bn=False):
        super(Master_noAtt, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        # self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_noAspp_decoder(num_classes, backbone, BatchNorm)
        self.project = nn.Sequential( 
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(2048, 512, bias=False),
                nn.ReLU(),
                nn.Linear(512, 128, bias=False),
            )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fla = nn.Flatten()
        self.lin = nn.Linear(256, 512)
        self.sig = nn.Sigmoid()

        self.norm_conv = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True),)

        self.last_conv = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                    #    nn.ReLU(),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                    #    nn.ReLU(),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.1))
        self.num_class = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.freeze_bn = freeze_bn

    def forward(self, input):
        if self.training:
            with torch.set_grad_enabled(not True):
                tmp = input.shape[0] // 3
                d, d0, d1, d2, d3, x = self.backbone(input, tmp)
                # x = self.aspp(x)
            x1 = x[:tmp,:,:,:]
            x2 = x[tmp:,:,:,:]
            y = self.project(x2)

            x = self.decoder(d0, d1, d2, d3, x1)

            x = F.interpolate(x, size=d.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, d], dim=1)
            x = self.norm_conv(x)
            x = self.last_conv(x)
            x = self.num_class(x)
            x = self.sig(x)

            x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
            return x, y
        else:
            d, d0, d1, d2, d3, x = self.backbone(input)
            # x = self.aspp(x)
            x = self.decoder(d0, d1, d2, d3, x)

            x = F.interpolate(x, size=d.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, d], dim=1)
            x = self.norm_conv(x)
            x = self.last_conv(x)
            x = self.num_class(x)
            x = self.sig(x)

            x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
            return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

class Master_noSelf(nn.Module):
    def __init__(self, sparse_radio=0.6, backbone='resnet50_ori', output_stride=16, num_classes=1,
                 sync_bn=True, freeze_bn=False):
        super(Master_noSelf, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        # self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_noAspp_decoder(num_classes, backbone, BatchNorm)
        self.project = nn.Sequential( 
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(256, 512, bias=False),
                nn.ReLU(),
                nn.Linear(512, 128, bias=False),
            )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fla = nn.Flatten()
        self.lin = nn.Linear(256, 512)
        self.att = Dilated10ConvAttentionMap1x1AvgTauWithSparsity(sparse_radio)
        self.sig = nn.Sigmoid()

        self.norm_conv = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True),)

        self.last_conv = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                     # nn.ReLU(),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                     # nn.ReLU(),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.1))
        self.num_class = nn.Conv2d(256, 1, kernel_size=1, stride=1)
        self.num_class_new = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.freeze_bn = freeze_bn

    def forward(self, input):       
        with torch.set_grad_enabled(not True):
            z1, z2 = self.att(input)
            d, d0, d1, d2, d3, x = self.backbone(z1)
            # x = self.aspp(x)
        x = self.decoder(d0, d1, d2, d3, x)

        x = F.interpolate(x, size=d.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, d], dim=1)
        x = self.norm_conv(x)
        x = self.last_conv(x)
        x = self.num_class_new(x)
        x = self.sig(x)

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x, z2, z1
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

class Master_noSelfnoAtt(nn.Module):
    def __init__(self, backbone='resnet50_ori', output_stride=16, num_classes=1,
                 sync_bn=True, freeze_bn=False):
        super(Master_noSelfnoAtt, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        # self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_noAspp_decoder(num_classes, backbone, BatchNorm)
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fla = nn.Flatten()
        self.lin = nn.Linear(256, 512)
        self.sig = nn.Sigmoid()

        self.norm_conv = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True),)

        self.last_conv = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                     # nn.ReLU(),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                     # nn.ReLU(),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.1))
        
        self.num_class = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.freeze_bn = freeze_bn

    def forward(self, input):
        with torch.set_grad_enabled(not True):
            d, d0, d1, d2, d3, x = self.backbone(input)
            # x = self.aspp(x)
        x = self.decoder(d0, d1, d2, d3, x)

        x = F.interpolate(x, size=d.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, d], dim=1)
        x = self.norm_conv(x)
        x = self.last_conv(x)
        x = self.num_class(x)
        x = self.sig(x)

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

if __name__ == "__main__":
    model = Dilated10ConvAttentionMap1x1AvgTauWithSparsity(0.6)
    model = Master_all(0.6, backbone='resnet50', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 256, 256)
    output = model(input)
    summary(model, [(3, 256, 256)])
    print(output.size())