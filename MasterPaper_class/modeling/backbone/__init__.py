from modeling.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet101':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'resnet50':
        return resnet.resnet50(output_stride, BatchNorm)
    elif backbone == 'resnet50_ori':
        return resnet.resnet50_ori(output_stride, BatchNorm)
    elif backbone == 'resnet18':
        return resnet.resnet18(output_stride, BatchNorm)
    elif backbone == 'resnet34':
        return resnet.resnet34(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
