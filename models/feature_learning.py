import torch
from torch import nn
import math
from .hiarachical_layers import FiLMLayer, FiLMDownsample


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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


class ResNetbasic(nn.Module):

    def __init__(self, block, layers, z_dim, num_classes=1000):
        self.inplanes = 64
        self.z_dim = z_dim
        super(ResNetbasic, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, dropout=-1):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class BasicBlockFiLM(nn.Module):
    """ResNet BasicBlock with FiLM conditioning instead of domain-specific BN."""
    expansion = 1

    def __init__(self, inplanes, planes, emb_dim, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.film1 = FiLMLayer(planes, emb_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.film2 = FiLMLayer(planes, emb_dim)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, d_emb):
        residual = x

        out = self.conv1(x)
        out = self.film1(out, d_emb)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.film2(out, d_emb)

        if self.downsample is not None:
            residual = self.downsample(x, d_emb)

        out += residual
        out = self.relu(out)

        return out


class FiLMSequential(nn.Module):
    """Sequential container that passes d_emb to every child module."""
    def __init__(self, *modules):
        super().__init__()
        self.blocks = nn.ModuleList(modules)

    def forward(self, x, d_emb):
        for block in self.blocks:
            x = block(x, d_emb)
        return x


class ResNetFiLM(nn.Module):
    """ResNet backbone with FiLM domain conditioning. Accepts batch domain_ids."""

    def __init__(self, layers, z_dim, num_domains=4, emb_dim=32):
        super().__init__()
        self.inplanes = 64
        self.z_dim = z_dim
        self.emb_dim = emb_dim
        self.domain_emb = nn.Embedding(num_domains, emb_dim)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.film1 = FiLMLayer(64, emb_dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], emb_dim)
        self.layer2 = self._make_layer(128, layers[1], emb_dim, stride=2)
        self.layer3 = self._make_layer(256, layers[2], emb_dim, stride=2)
        self.layer4 = self._make_layer(512, layers[3], emb_dim, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks, emb_dim, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlockFiLM.expansion:
            downsample = FiLMDownsample(
                self.inplanes, planes, BasicBlockFiLM.expansion, stride, emb_dim)

        layer_list = []
        layer_list.append(BasicBlockFiLM(self.inplanes, planes, emb_dim, stride, downsample))
        self.inplanes = planes * BasicBlockFiLM.expansion
        for _ in range(1, blocks):
            layer_list.append(BasicBlockFiLM(self.inplanes, planes, emb_dim))

        return FiLMSequential(*layer_list)

    def forward(self, x, domain_ids):
        """
        Args:
            x: (B, 1, H, W) input images
            domain_ids: (B,) integer domain indices
        """
        d_emb = self.domain_emb(domain_ids)
        x = self.conv1(x)
        x = self.film1(x, d_emb)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, d_emb)
        x = self.layer2(x, d_emb)
        x = self.layer3(x, d_emb)
        x = self.layer4(x, d_emb)

        return x
