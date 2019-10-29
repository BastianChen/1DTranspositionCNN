from auto_encoder.trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer("../auto_encoder/models", "net.pth", "../auto_encoder/datasets", "../auto_encoder/image")
    trainer.train()
