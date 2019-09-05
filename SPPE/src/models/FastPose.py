import torch.nn as nn
from torch.autograd import Variable

from .layers.SE_Resnet import SEResnet
from .layers.DUC import DUC


def createModel():
    return FastPose(nClasses=33)


class FastPose(nn.Module):
    DIM = 128
    opt_nClasses = 33

    def __init__(self, nClasses=opt_nClasses):
        super(FastPose, self).__init__()

        self.preact = SEResnet('resnet101')

        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2)
        self.duc2 = DUC(256, 512, upscale_factor=2)
        self.nClasses = nClasses

        self.conv_out = nn.Conv2d(
            self.DIM, self.nClasses, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Variable):
        out = self.preact(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)

        out = self.conv_out(out)
        return out
