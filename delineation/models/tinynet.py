from delineation.models.blocks.basic import *


class TinyNet(nn.Module):

    def __init__(self, n_channel):
        super().__init__()

        self.dconv_down1 = double_conv(n_channel, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 64)
        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.dconv_down1(x)
        x = self.dconv_down2(x)
        x = self.dconv_down3(x)
        out = self.conv_last(x)
        return out, x