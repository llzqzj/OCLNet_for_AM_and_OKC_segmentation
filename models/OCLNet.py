import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
# from torchsummary import summary


class OCL(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=64):
        """
        Implementations based on the OCL paper: https://arxiv.org/abs/1606.06650
        """

        super(OCL, self).__init__()

        features = init_features
        self.encoder1 = OCL._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = OCL._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = OCL._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = OCL._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = OCL._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16 + out_channels * 0, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = OCL._block((features * 8) * 2 , features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8 + out_channels * 1, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = OCL._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4 + out_channels * 2, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = OCL._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2 + out_channels * 3, features, kernel_size=2, stride=2
        )
        self.decoder1 = OCL._block(features * 2, features, name="dec1")

        self.pred5 = nn.Conv3d(features * 16, out_channels, kernel_size = 1)
        self.pred4 = nn.Conv3d(features * 8, out_channels, kernel_size = 1)
        self.pred3 = nn.Conv3d(features * 4, out_channels, kernel_size = 1)
        self.pred2 = nn.Conv3d(features * 2, out_channels, kernel_size = 1)

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        # p5 = self.pred5(bottleneck)
        # p5up4 = nn.functional.interpolate(p5, size = list(enc4.shape)[2:], mode = 'trilinear')
        # p5up3 = nn.functional.interpolate(p5, size = list(enc3.shape)[2:], mode = 'trilinear')
        # p5up2 = nn.functional.interpolate(p5, size = list(enc2.shape)[2:], mode = 'trilinear')

        # bottleneck_cat = torch.cat((bottleneck, p5), dim=1)
        # dec4 = self.upconv4(bottleneck_cat)
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        p4 = self.pred4(dec4)
        p4up3 = nn.functional.interpolate(p4, size = list(enc3.shape)[2:], mode = 'trilinear')
        p4up2 = nn.functional.interpolate(p4, size = list(enc2.shape)[2:], mode = 'trilinear')

        # dec4_cat = torch.cat((dec4, p4, p5up4), dim=1)
        dec4_cat = torch.cat((dec4, p4), dim=1)
        dec3 = self.upconv3(dec4_cat)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        p3 = self.pred3(dec3)
        p3up2 = nn.functional.interpolate(p3, size = list(enc2.shape)[2:], mode = 'trilinear')

        # dec3_cat = torch.cat((dec3, p3, p5up3, p4up3), dim=1)
        dec3_cat = torch.cat((dec3, p3, p4up3), dim=1)
        dec2 = self.upconv2(dec3_cat)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        p2 = self.pred2(dec2)

        # dec2_cat = torch.cat((dec2, p2, p5up2, p4up2, p3up2), dim=1)
        dec2_cat = torch.cat((dec2, p2, p4up2, p3up2), dim=1)
        dec1 = self.upconv1(dec2_cat)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        outputs = self.conv(dec1)
        return outputs
        # return outputs, p2, p3, p4
        # return outputs, p2, p3, p4, p5

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )