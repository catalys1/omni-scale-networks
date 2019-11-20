'''Slightly modified implementation of OSNet from the paper
"Omni-Scale Feature Learning for Person Re-Identification" by
Kaiyang Zhou, Yongxin Yang, Andrea Cavallaro, Tao Xiang
(https://arxiv.org/abs/1905.00953)

Author: Connor Anderson
'''
import torch
import torchvision


__all__ = ['OSNet']


def passthrough(x):
    '''Noop layer'''
    return x


def conv1x1(inc, outc, linear=False):
    '''1x1 conv -> batchnorm -> (optional) ReLU'''
    layers = [torch.nn.Conv2d(inc, outc, 1, bias=False),
              torch.nn.BatchNorm2d(outc)]
    if not linear:
        layers.append(torch.nn.ReLU(inplace=True))
    return torch.nn.Sequential(*layers)


def conv3x3(inc, outc, stride=1):
    '''3x3 conv -> batchnorm -> ReLU'''
    return torch.nn.Sequential(
        torch.nn.Conv2d(inc, outc, 3, padding=1, stride=stride, bias=False),
        torch.nn.BatchNorm2d(outc),
        torch.nn.ReLU(inplace=True)
    )


def convlite(inc, outc):
    '''Lite conv layer. 1x1 conv -> 3x3 depthwise conv -> batchnorm -> ReLU'''
    return torch.nn.Sequential(
        torch.nn.Conv2d(inc, outc, 1, bias=False),
        torch.nn.Conv2d(outc, outc, 3, padding=1, groups=outc, bias=False),
        torch.nn.BatchNorm2d(outc),
        torch.nn.ReLU(inplace=True)
    )


def convstack(inc, outc, n=1):
    '''A stack of n convlite layers'''
    convs = convlite(inc, outc)
    if n > 1:
        convs = [convs] + [convlite(outc, outc) for i in range(n-1)]
        convs = torch.nn.Sequential(*convs)
    return convs


class Gate(torch.nn.Module):
    '''Unified Aggregation Gate.
    
    Args:
        c (int): number of channels (input and output are the same)
    '''
    def __init__(self, c):
        super().__init__()
        self.gate = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(c, c//16, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(c//16, c, 1),
            torch.nn.Sigmoid())
    
    def forward(self, x):
        g = self.gate(x)
        x = x * g
        return x


class Bottleneck(torch.nn.Module):
    '''OSNet bottleneck layer (figure 4 in the paper).

    Args:
        inc (int): number of input feature channels
        outc (int): number of output feature channels
        t (int): number of scales
        reduction (int): factor to reduce/expand the number of feature
            channels before/after multiscale layers
    '''
    def __init__(self, inc, outc, t=4, reduction=4):
        super().__init__()
        midc = inc // reduction
        self.conv1 = conv1x1(inc, midc)
        self.streams = torch.nn.ModuleList([
            convstack(midc, midc, n=i+1) for i in range(t)
        ])
        self.gate = Gate(midc)
        self.conv2 = conv1x1(midc, outc, linear=True)
        self.project = (passthrough if inc == outc else
                        conv1x1(inc, outc, linear=True))

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = [s(x) for s in self.streams]
        x = sum([self.gate(xi) for xi in x])
        x = self.conv2(x)
        x = self.project(identity) + x
        x = torch.nn.functional.relu(x, inplace=True)
        return x


class OSNet(torch.nn.Module):
    '''OmniScale network.

    Args:
        n_class (int): number of classes for classification
    '''
    def __init__(self, n_class):
        super().__init__()
        # replace the 7x7 with 3 3x3s
        self.conv1 = torch.nn.Sequential(
            conv3x3(3, 32, stride=2),
            conv3x3(32, 32),
            conv3x3(32, 64))
        self.maxpool = torch.nn.MaxPool2d(3, 2)
        self.conv2 = torch.nn.Sequential(
            Bottleneck(64, 256),
            Bottleneck(256, 256),
            conv1x1(256, 256),
            torch.nn.AvgPool2d(2, 2))
        self.conv3 = torch.nn.Sequential(
            Bottleneck(256, 384),
            Bottleneck(384, 384),
            conv1x1(384, 384),
            torch.nn.AvgPool2d(2, 2))
        self.conv4 = torch.nn.Sequential(
            Bottleneck(384, 512),
            Bottleneck(512, 512),
            conv1x1(512, 512),
            torch.nn.AvgPool2d(2, 2))
        # add extra avg pool and extra 1x1 conv
        self.conv5 = conv1x1(512, 512)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        # replace the extra fc (512 x 512) with a single classifier
        self.fc = torch.nn.Linear(512, n_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

