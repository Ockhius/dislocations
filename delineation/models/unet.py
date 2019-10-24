from delineation.models.blocks.basic import *


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(n_class, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv_drop(256, 512)
        self.dconv_down5 = double_conv_drop(512, 1024)

        self.down = down_sample()

        self.up4 = up_sample(1024, 512)
        self.up3 = up_sample(512, 256)
        self.up2 = up_sample(256, 128)
        self.up1 = up_sample(128, 64)

        self.dconv_up4 = double_conv(1024, 512)
        self.dconv_up3 = double_conv(512, 256)
        self.dconv_up2 = double_conv(256, 128)
        self.dconv_up1 = double_conv(128, 64)

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.down(conv1)

        conv2 = self.dconv_down2(x)
        x = self.down(conv2)

        conv3 = self.dconv_down3(x)
        x = self.down(conv3)

        conv4 = self.dconv_down4(x)
        x = self.down(conv4)

        x = self.dconv_down5(x)

        x = self.up4(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)

        x = self.up3(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.up2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)

        x = self.up1(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)
        return out, x