from delineation.models.blocks.basic import *


class UNet_small(nn.Module):

    def __init__(self, n_channel):
        super().__init__()

        self.dconv_down1 = double_conv(n_channel, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv_drop(128, 256)

        self.down = down_sample()

        self.up2 = up_sample(256, 128)
        self.up1 = up_sample(128, 64)

        self.dconv_up2 = double_conv(256, 128)
        self.dconv_up1 = double_conv(128, 64)

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.down(conv1)

        conv2 = self.dconv_down2(x)
        x = self.down(conv2)

        x = self.dconv_down3(x)

        x = self.up2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)

        x = self.up1(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)
        return out, x