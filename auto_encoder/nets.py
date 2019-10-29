import torch
from torch import nn

'''自编码网络模型'''


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            ConvolutionalLayer(1, 128, 3, 2, 1),  # n,128,14,14
            ConvolutionalLayer(128, 512, 3, 2, 1)  # n,512,7,7
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(512 * 7 * 7, 128),
            nn.Sigmoid()
        )

    def forward(self, data):
        conv_layer = self.conv_layer(data)
        data = conv_layer.reshape(data.size(0), 512 * 7 * 7)
        output = self.linear_layer(data)
        return output


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(128, 512 * 7 * 7),
            nn.BatchNorm1d(512 * 7 * 7),
            nn.PReLU()
        )
        self.conv_transpose_layer = nn.Sequential(
            ConvTransposeLayer(512, 128, 3, 2, 1, 1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            ConvTransposeLayer(128, 1, 3, 2, 1, 1),
            nn.PReLU()
        )

    def forward(self, data):
        linear_layer = self.linear_layer(data)
        linear_layer = linear_layer.reshape(-1, 512, 7, 7)
        output = self.conv_transpose_layer(linear_layer)
        return output


class Total_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, data):
        encoder = self.encoder(data)
        decoder = self.decoder(encoder)
        return decoder


# 封装卷积层
class ConvolutionalLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, groups=groups, bias=bias),
            nn.BatchNorm2d(output_channels),
            nn.PReLU()
        )

    def forward(self, data):
        return self.layer(data)


# 封装反卷积
class ConvTransposeLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, output_padding, bias=bias)
        )

    def forward(self, data):
        return self.layer(data)


if __name__ == '__main__':
    input = torch.Tensor(5, 1, 28, 28)
    net = Total_Net()
    output = net(input)
    print(output)
    print(output.shape)
