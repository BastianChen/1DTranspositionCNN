from cnn_1d.nets import ConvNet, LstmNet
from cnn_1d.dataset import Dataset
import matplotlib.pyplot as plt
import torch


class Detector:
    def __init__(self, net_path, dataset_path, isLstm=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.isLstm = isLstm
        if isLstm:
            self.net = LstmNet().to(self.device)
        else:
            self.net = ConvNet().to(self.device)
        self.data = Dataset(dataset_path).getData()
        self.net.load_state_dict(torch.load(net_path))
        self.input_list = []
        self.output_list = []
        self.net.eval()

    def detect(self):
        self.input_list.extend(self.data[len(self.data) - 9:])
        for _ in range(60):
            if self.isLstm:
                last_data = torch.tensor(self.input_list).reshape(-1, 9, 1).to(self.device)
            else:
                last_data = torch.tensor(self.input_list).reshape(-1, 1, 9).to(self.device)
            output = self.net(last_data)
            self.input_list.append(output.item())
            self.output_list.append(output.item())
            self.input_list.pop(0)
            # print(output.item())
        print(self.output_list)
        print(self.data)
        plt.plot(self.output_list)
        plt.plot(self.data)
        plt.legend(['prediction', 'real'], loc='upper right')
        plt.show()


if __name__ == '__main__':
    # detect = Detector("../cnn_1d/models/net.pth", "../cnn_1d/data/data2.xlsx", isLstm=False)
    detect = Detector("../cnn_1d/models/lstm_net.pth", "../cnn_1d/data/data2.xlsx", isLstm=True)
    detect.detect()
