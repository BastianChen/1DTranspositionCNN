import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from auto_encoder.nets import Total_Net
import os
from torchvision.utils import save_image


class Trainer:
    def __init__(self, save_net_path, net_name, dataset_path, save_img_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_net_path
        self.net_name = net_name
        self.save_img_path = save_img_path
        self.net = Total_Net().to(self.device)
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.train_data = DataLoader(datasets.MNIST(dataset_path, train=True, download=False, transform=self.trans),
                                     batch_size=100, shuffle=True)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters())
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        else:
            self.net.load_state_dict(torch.load(self.save_path))
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)

    def train(self):
        epoch = 1
        loss_new = 100
        while True:
            for i, (image, label) in enumerate(self.train_data):
                image, label = image.to(self.device), label.to(self.device)
                output = self.net(image)
                loss = self.loss_fn(output, image)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if i % 10 == 0:
                    print("epoch:{},i:{},loss:{}".format(epoch, i, loss.item()))
                    image = image.detach()
                    output = output.detach()
                    save_image(image, "{}/{}-{}-input_image.jpg".format(self.save_img_path, epoch, i), nrow=10)
                    save_image(output, "{}/{}-{}-ouput_image.jpg".format(self.save_img_path, epoch, i), nrow=10)

                if loss.item() < loss_new:
                    loss_new = loss.item()
                    torch.save(self.net.state_dict(), os.path.join(self.save_path, self.net_name))
            epoch += 1
