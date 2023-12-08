import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # This sequential double convolution is seen and repeated throughout the UNET
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), # Kernel is 3, stride is 1, padding is 1
            nn.BatchNorm2d(out_channels), # if bias is true true then this batchnorm can be removed.
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.doubleconv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],): 
        super(UNET, self).__init__()
        # ups and downs will contain the actions that our model will have to take.
        self.uphill = nn.ModuleList()
        self.downhill = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downhill.append(DoubleConv(in_channels, feature)) #in_channel -> out_channel
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            # This is the step when we are going up one level and doubleing our inputs 
            self.uphill.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            # This is the step when we are applying 2 conv layers within that level
            self.uphill.append(DoubleConv(feature*2, feature))
        
        # This is the lowest point of the UNET
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # This is the final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # print("x.shape is ", x.shape) #x .shape is  torch.Size([16, 3, 160, 240])
        skip_connections = []

        for down in self.downhill:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.uphill), 2):
            x = self.uphill[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                # x = TF.resize(x, size=skip_connection.shape[2:])
                x = TF.resize(x, size=skip_connection.shape[2:], antialias=True)


            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.uphill[idx+1](concat_skip)
        x = self.final_conv(x)

        return x

def test():
    x = torch.randn((3, 3, 140, 140))
    model = UNET(in_channels=3, out_channels=1)
    preds = model(x)
    print(x.shape)
    print(preds.shape)
    assert preds.shape == (3, 1, 140, 140)

if __name__ == "__main__":
    test()