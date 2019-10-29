import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from auto_encoder.nets import Total_Net
import os
from torchvision.utils import save_image


class Detector:
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
        self.test_data = DataLoader(datasets.MNIST(dataset_path, train=False, download=False, transform=self.trans),
                                    batch_size=100, shuffle=True)
        self.net.load_state_dict(torch.load(os.path.join(self.save_path, self.net_name)))
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)
        self.net.eval()

    def detect(self):
        epoch = 1
        while True:
            for i, (image, label) in enumerate(self.test_data):
                image, label = image.to(self.device), label.to(self.device)
                output = self.net(image)
                image = image.detach()
                output = output.detach()
                save_image(image, "{}/{}-{}-input_image.jpg".format(self.save_img_path, epoch, i), nrow=10)
                save_image(output, "{}/{}-{}-ouput_image.jpg".format(self.save_img_path, epoch, i), nrow=10)
            epoch += 1


if __name__ == '__main__':
    detector = Detector("../auto_encoder/models", "net.pth", "../auto_encoder/datasets", "../auto_encoder/image_test")
    detector.detect()
