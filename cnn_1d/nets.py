from torch import nn
import torch


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            ConvolutionalLayer(1, 128, 3, 2, 1),  # n,128,5
            ConvolutionalLayer(128, 512, 3, 2, 1),  # n,512,3
            ConvolutionalLayer(512, 1024, 3)  # n,1024,1
        )
        self.liner_layer = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        conv = self.conv_layer(data)
        conv = conv.reshape(-1, 1024)
        output = self.liner_layer(conv)
        return output


class LstmNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_layer = nn.LSTM(1, 1024, 2, batch_first=True)
        self.liner_layer = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        lstm_layer, (h, c) = self.lstm_layer(data)
        lstm_layer = lstm_layer[:, -1]
        lstm_layer = lstm_layer.reshape(-1, 1024)
        output = self.liner_layer(lstm_layer)
        return output


class ConvolutionalLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size, stride, padding, groups=groups, bias=bias),
            # nn.BatchNorm1d(output_channels),
            nn.PReLU()
        )

    def forward(self, data):
        return self.layer(data)


if __name__ == '__main__':
    # net = LstmNet()
    net = ConvNet()
    data = torch.tensor([x / 9 for x in range(9)]).reshape(1, 1, 9)
    # data = torch.tensor([x / 9 for x in range(9)]).reshape(1, 9, 1)
    output = net(data)
    print(output)
    print(output.shape)
