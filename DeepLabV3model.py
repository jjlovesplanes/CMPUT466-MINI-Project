import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# ASPP (Atrous Spatial Pyramid Pooling) Module
class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(ASPPModule, self).__init__()
        # Atrous Convolutional Layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False)
        # Batch Normalization
        self.bn = nn.BatchNorm2d(out_channels)
        # ReLU Activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Forward pass through ASPP Module
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# DeepLabV3 Model
class DeepLabV3(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=256):
        super(DeepLabV3, self).__init__()

        # Backbone (A series of Convolutional Layers)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )

        # ASPP Module (Atrous Spatial Pyramid Pooling)
        self.aspp = nn.ModuleList([
            ASPPModule(features, features, kernel_size=1, padding=0, dilation=1),
            ASPPModule(features, features, kernel_size=3, padding=6, dilation=6),
            ASPPModule(features, features, kernel_size=3, padding=12, dilation=12),
            ASPPModule(features, features, kernel_size=3, padding=18, dilation=18),
        ])

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Final Convolutional Layer
        self.final_conv = nn.Conv2d(features * 5, out_channels, kernel_size=1)

    def forward(self, x):
        # Forward pass through the backbone
        x = self.backbone(x)

        # Forward pass through each ASPP module and concatenate the results
        aspp_out = [aspp_module(x) for aspp_module in self.aspp]
        aspp_out = torch.cat(aspp_out, dim=1)

        # Global Average Pooling
        global_avg_pool_out = self.global_avg_pool(x)
        global_avg_pool_out = torch.repeat_interleave(global_avg_pool_out, x.size(2), dim=2)
        global_avg_pool_out = torch.repeat_interleave(global_avg_pool_out, x.size(3), dim=3)

        # Concatenate ASPP output and Global Average Pooling output
        x = torch.cat([aspp_out, global_avg_pool_out], dim=1)
        # Final Convolution to get the segmentation map
        x = self.final_conv(x)

        return x

# Testing the model
def test():
    # Create a random input tensor
    x = torch.randn((3, 3, 140, 140))
    model = DeepLabV3(in_channels=3, out_channels=1)
    preds = model(x)
    print(x.shape)
    print(preds.shape)
    assert preds.shape == (3, 1, 140, 140)

# Run the test when the script is executed
if __name__ == "__main__":
    test()
