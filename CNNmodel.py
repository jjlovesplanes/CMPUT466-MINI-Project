import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        
        # The final convolutional layer to produce the output
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
  
        x = self.final_conv(x)
        
        return x

def test():
    x = torch.randn((3, 3, 140, 140))  
    model = CNN(in_channels=3, out_channels=1) 
    preds = model(x)
    print(x.shape)
    print(preds.shape)
    assert preds.shape == (3, 1, 140, 140)

if __name__ == "__main__":
    test()
