import torch
import torch.nn as nn
import torchvision.models as models


class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(
                scale_factor=scale_factor, mode="bilinear", align_corners=False
            ),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor):
        # Check if the width dimension is odd and needs zero padding
        x = self.upsample_layer(x)

        if x.shape[-1] != x_skip.shape[-1] or x.shape[-2] != x_skip.shape[-2]:
            x = nn.functional.interpolate(
                x, size=(x_skip.shape[-2], x_skip.shape[-1]), mode="bilinear"
            )

        return x + x_skip


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, dropout_rate: float = 0.0):
        super(SegmentationHead, self).__init__()

        backbone = models.resnet18(pretrained=False, zero_init_residual=True)

        self.first_conv = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # Upsampling layers
        self.up3_skip = UpsamplingAdd(
            in_channels=256, out_channels=128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(
            in_channels=128, out_channels=64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(
            in_channels=64, out_channels=in_channels, scale_factor=2)

        # Segmentation head
        self.dropout = nn.Dropout(
            dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Conv2d(in_channels, n_classes, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor):
        # (H, W)
        skip_x = {"1": x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # (H/4, W/4)
        x = self.layer1(x)
        skip_x["2"] = x
        x = self.dropout(x)

        x = self.layer2(x)
        skip_x["3"] = x
        x = self.dropout(x)

        # (H/8, W/8)
        x = self.layer3(x)
        x = self.dropout(x)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x["3"])
        x = self.dropout(x)

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x["2"])
        x = self.dropout(x)

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x["1"])
        x = self.dropout(x)

        segmentation_output = self.segmentation_head(x)

        return segmentation_output
