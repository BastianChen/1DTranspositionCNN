import torch
from torch import nn
from cnn_1d.nets import ConvNet, LstmNet
from cnn_1d.dataset import Dataset
import os


class Trainer:
    def __init__(self, save_path, dataset_path, isLstm=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path
        self.isLstm = isLstm
        self.train_data = Dataset(dataset_path).getData()
        if isLstm:
            self.net = LstmNet().to(self.device)
        else:
            self.net = ConvNet().to(self.device)
        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        else:
            self.net.load_state_dict(torch.load(self.save_path))

    def train(self):
        epoch = 1
        loss_new = 100
        while True:
            for i in range(self.train_data.shape[0] - 9):
                if self.isLstm:
                    input = torch.Tensor(self.train_data[i:i + 9].reshape(-1, 9, 1)).to(self.device)
                else:
                    input = torch.Tensor(self.train_data[i:i + 9].reshape(-1, 1, 9)).to(self.device)
                label = torch.Tensor(self.train_data[i + 9: i + 10]).to(self.device)
                output = self.net(input)
                loss = self.loss_fn(output, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % 10 == 0:
                    print("epoch:{},i:{},loss:{},output:{},label:{},output-label:{:.6f}".format(epoch, i, loss.item(),
                                                                                                output.item(),
                                                                                                label.item(),
                                                                                                output.item() - label.item()))
                if loss.item() < loss_new:
                    loss_new = loss.item()
                    torch.save(self.net.state_dict(), self.save_path)
            epoch += 1
