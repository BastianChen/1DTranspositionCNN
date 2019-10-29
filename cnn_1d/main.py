from cnn_1d.trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer("../cnn_1d/models/net.pth", "../cnn_1d/data/data2.xlsx", isLstm=False)
    # trainer = Trainer("../cnn_1d/models/lstm_net.pth", "../cnn_1d/data/data2.xlsx", isLstm=True)
    trainer.train()
