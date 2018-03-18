import torch.nn as nn
import torch
from torch.autograd import Variable
import math
import torchvision.models as models
import sys
sys.path.append("../")

import utils.utils as utils

class ScaleLayer(nn.Module):

    def __init__(self, init_value=1e-3):
        """
        Adopted from https://discuss.pytorch.org/t/is-scale-layer-available-in-pytorch/7954/6
        """
        super(ScaleLayer, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input_data):
        return input_data * self.scale

class CropLayer(nn.Module):

    def __init__(self):
        super(CropLayer, self).__init__()

    def forward(self, input_data, offset):
        """
        Currently, only for specific axis, the same offset. Assume for h, w dim.
        """
        cropped_data = input_data[:, :, offset:-offset, offset:-offset]
        return cropped_data

class SliceLayer(nn.Module):

    def __init__(self):
        super(SliceLayer, self).__init__()

    def forward(self, input_data):
        """
        slice into several sinle piece in a specific dimension. Here for dim=1
        """
        sliced_list = []
        for idx in xrange(input_data.size()[1]):
            sliced_list.append(input_data[:, idx, :, :].unsqueeze(1))

        return sliced_list

class ConcatLayer(nn.Module):

    def __init__(self):
        super(ConcatLayer, self).__init__()

    def forward(self, input_data_list, dim):
        concat_feats = torch.cat((input_data_list), dim=dim)
        return concat_feats

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, special_case=False):
        """
        special case only for res5a branch
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.scale_conv1 = ScaleLayer()

        if special_case:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   dilation=4, padding=4, bias=False)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               dilation=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.scale_conv2 = ScaleLayer()

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.scale_conv3 = ScaleLayer()

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.scale_downsample = ScaleLayer()
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.scale_conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.scale_conv2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.scale_conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            residual = self.scale_downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=20):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.scale_conv1 = ScaleLayer()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, special_case=True) # Notice official resnet is 2, but CASENet here is 1.

        # Added by CASENet to get feature map from each branch in different scales.
        self.score_side1 = nn.Conv2d(64, 1, kernel_size=1, bias=False)
        self.score_side2 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, bias=False),
            nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, bias=False)
        ) # PyTorch currently does not have crop layer, so we use index to crop in forward.
        
        self.score_side3 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, bias=False),
            nn.ConvTranspose2d(1, 1, kernel_size=8, stride=4, bias=False)
        ) # PyTorch currently does not have crop layer, so we use index to crop in forward.
        
        self.score_side5 = nn.Sequential(
            nn.Conv2d(2048, num_classes, kernel_size=1, bias=False),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)
        ) # PyTorch currently does not have crop layer, so we use index to crop in forward.

        self.score_fusion = nn.Conv2d(num_classes, num_classes, kernel_size=1)

        # Define crop, concat layer
        self.crop_layer = CropLayer()
        self.slice_layer = SliceLayer()
        self.concat_layer = ConcatLayer()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, special_case=False):
        """
        special case only for res5a branch
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, special_case=special_case))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, special_case=special_case))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) # BS X 64 X 352 X 352
        score_feats1 = self.score_side1(x)
        
        x = self.maxpool(x)

        x = self.layer1(x)
        score_feats2 = self.score_side2(x)
        cropped_score_feats2 = self.crop_layer(score_feats2, offset=1)

        x = self.layer2(x)
        score_feats3 = self.score_side3(x)
        cropped_score_feats3 = self.crop_layer(score_feats3, offset=2)
        
        x = self.layer3(x)
        x = self.layer4(x)
        score_feats5 = self.score_side5(x)
        cropped_score_feats5 = self.crop_layer(score_feats5, offset=4) # BS X 20 X 352 X 352. The output of it will be used to get a loss for this branch.
        sliced_list = self.slice_layer(cropped_score_feats5) # Each element is BS X 1 X 352 X 352
        concat_feats = self.concat_layer(sliced_list, dim=1) # BS X 20 X 352 X 352
        fused_feats = self.score_fusion(concat_feats) # BS X 20 X 352 X 352. The output of this will gen loss for this branch. So, totaly 2 loss. (same loss type)
        
        return cropped_score_feats5, fused_feats


def CASENet_resnet101(pretrained=False, num_classes=20):
    """Constructs a modified ResNet-101 model for CASENet.
    Args:
        pretrained (bool): If True, returns a model pre-trained on MSCOCO
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    if pretrained:
        utils.load_official_pretrained_model(model, "/ais/gobi5/jiaman/torch/models/resnet101-5d3b4d8f.pth") 
    return model

if __name__ == "__main__":
    model = CASENet_resnet101(pretrained=True, num_classes=20)
    input_data = torch.rand(2, 3, 352, 352)
    input_var = Variable(input_data)
    output1, output2  = model(input_var) 
    print("output1.size:{0}".format(output1.size()))
    print("output2.size:{0}".format(output2.size()))
