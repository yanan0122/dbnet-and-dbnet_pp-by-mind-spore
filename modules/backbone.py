from ctypes import util
import mindspore.nn as nn
from mindspore.common.initializer import Normal
from mindspore import load_checkpoint, load_param_into_net
from mindspore.common.initializer import HeNormal

import math
import sys
import mindspore as ms

sys.path.insert(0, '.')


# from utils.dcn import DeformConv2d

# set initializer to constant for debugging.
def conv3x3(inplanes, outplanes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, pad_mode="pad",
                     padding=1, weight_init=HeNormal())


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(BasicBlock, self).__init__()

        # print("BasicBlock out_channels", inplanes, planes)

        # set initializer to constant for debugging.
        self.with_dcn = dcn is not None

        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU()

        if self.with_dcn:

            from mindspore.ops import deformable_conv2d
            from mindspore import Tensor, Parameter
            from mindspore.common.initializer import HeNormal

            deformable_groups = dcn.get('deformable_groups', 1)

            offset_channels = 27

            # self.conv2 = conv3x3(planes, planes)

            self.conv2_weight = Parameter(Tensor(shape=(planes, planes, 3, 3), dtype=ms.float32, init=HeNormal))
            self.conv2_offset = nn.Conv2d(planes, offset_channels * deformable_groups, kernel_size=3, stride=1,
                                          pad_mode="pad",
                                          padding=1, weight_init=HeNormal(), has_bias=True)

            self.DCN = deformable_conv2d

        else:
            self.conv2 = conv3x3(planes, planes)

        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

        self.stride = stride

    def construct(self, x):
        residual = x
        # print("进入block")
        # print(x.shape)

        out = self.conv1(x)
        # print("=================== after block conv  ======================")
        # print(out[0][0][0][:5])

        out = self.bn1(out)
        # print("=================== after block bn1  ======================")
        # print(out[0][0][0][:5])

        out = self.relu(out)

        if self.with_dcn:

            conv2_offset = self.conv2_offset(out)

            out = self.DCN(out, self.conv2_weight, conv2_offset,
                           kernel_size=(3, 3), strides=(1, 1, 1, 1), padding=(1, 1, 1, 1))

        else:

            out = self.conv2(out)
            # print("conv2 ",out[0][0][0][:5])

        out = self.bn2(out)
        # print("bn2 ",out[0][0][0][:5])

        if self.downsample is not None:
            residual = self.downsample(x)
        # print(out.shape)
        # print(residual.shape)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(Bottleneck, self).__init__()
        self.with_dcn = dcn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False)

        # self.bn1 = BatchNorm2d(planes)
        self.bn1 = nn.BatchNorm2d(planes)

        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            from mindspore import Tensor, Parameter
            from mindspore.ops import deformable_conv2d
            from mindspore.common.initializer import HeNormal

            deformable_groups = dcn.get('deformable_groups', 1)

            offset_channels = 27

            # self.conv2 = conv3x3(planes, planes)

            self.conv2_weight = Parameter(Tensor(shape=(planes, planes, 3, 3), dtype=ms.float32, init=HeNormal))
            self.conv2_offset = nn.Conv2d(planes, offset_channels * deformable_groups, kernel_size=3, stride=1,
                                          pad_mode="pad", padding=1, weight_init=HeNormal(), has_bias=True)

            self.DCN = deformable_conv2d

        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, has_bias=False, pad_mode="pad",
                                   padding=1, stride=stride)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()

        self.downsample = downsample
        self.stride = stride
        self.dcn = dcn
        self.with_dcn = dcn is not None

    def construct(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.with_dcn:
            conv2_offset = self.conv2_offset(out)
            out = self.DCN(out, self.conv2_weight, conv2_offset,
                           kernel_size=(3, 3), strides=(1, 1, 1, 1), padding=(1, 1, 1, 1))

        else:
            out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Cell):

    def __init__(self, block, layers, num_classes=1000, dcn=None):

        self.inplanes = 64

        super(ResNet, self).__init__()

        # TODO: set initializer to constant for debugging.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode="pad",
                               has_bias=False, weight_init=HeNormal())  # same卷积，图片尺寸不变

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # TODO: pytorch maxpool2d sets padding=1 but mindspore maxpool2d can't. so just use pad_mode.
        self.maxpool = nn.SequentialCell([
              nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT"),
              nn.MaxPool2d(kernel_size=3, stride=2)])

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dcn=dcn)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dcn=dcn)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dcn=dcn)
        for m in self.cells():

            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight_init = Normal(0, math.sqrt(2. / n))

    def _make_layer(self, block, planes, blocks, stride=1, dcn=None):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample的功能是调整residual使其和out保持相同尺寸，out的变化由plane和stride控制
            downsample = nn.SequentialCell(
                # set initializer to constant for debugging.
                nn.Conv2d(self.inplanes, planes * block.expansion, pad_mode="pad",
                          kernel_size=1, stride=stride, has_bias=False, weight_init=HeNormal()),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes,
                            stride, downsample, dcn=dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcn))

        return nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.conv1(x)
        # print("=================== 1  ======================")
        # print(x[0][0][0][:5])
        x = self.bn1(x)
        # print("=================== 2  ======================")
        # print(x[0][0][0][:5])
        x = self.relu(x)
        # print("=================== 3  ======================")
        # print(x[0][0][0][:5])
        x = self.maxpool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x2, x3, x4, x5


def resnet18(pretrained=True, pretrained_ckpt=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained:
        ms_dict = load_checkpoint(pretrained_ckpt)
        param_not_load = load_param_into_net(model, ms_dict)
        print(param_not_load)

    return model


def deformable_resnet18(pretrained=True, pretrained_ckpt=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], dcn=dict(deformable_groups=1), **kwargs)

    if pretrained:
        ms_dict = load_checkpoint(pretrained_ckpt)
        param_not_load = load_param_into_net(model, ms_dict)
        print(param_not_load)

    return model


def resnet50(pretrained=True, pretrained_ckpt=None, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        ms_dict = load_checkpoint(pretrained_ckpt)
        param_not_load = load_param_into_net(model, ms_dict)
        print(param_not_load)

    return model


def deformable_resnet50(pretrained=True, pretrained_ckpt=None, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], dcn=dict(deformable_groups=1), **kwargs)

    if pretrained:
        ms_dict = load_checkpoint(pretrained_ckpt)
        param_not_load = load_param_into_net(model, ms_dict)
        print(param_not_load)

    return model


def get_backbone(initializer):
    backbone_dict = {
        "resnet18": resnet18,
        "deformable_resnet18": deformable_resnet18,
        "resnet50": resnet50,
        "deformable_resnet50": deformable_resnet50,
    }
    return backbone_dict[initializer]
