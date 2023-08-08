import timm
import torch
import torch.nn as nn
from torchsummary import summary


def set_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


class SiameseNetwork(nn.Module):
    """
    Siamese Network Implementation
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.3),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.3),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.Flatten(),
            nn.Linear(6144, 512),
        )

    def forward(self, image):
        embedding = self.net(image)
        return embedding


class EfficientNet(nn.Module):
    def __init__(self, emb_size=512, pretrained=False):
        super().__init__()

        self.efficientnet = timm.create_model(
            "efficientnet_b0", pretrained=False
        )
        self.efficientnet.classifier = nn.Linear(
            in_features=self.efficientnet.classifier.in_features,
            out_features=emb_size,
        )

    def forward(self, images):
        embeddings = self.efficientnet(images)
        return embeddings


class MobileNet(nn.Module):
    def __init__(self, emb_size=512, pretrained=False):
        super().__init__()

        self.mobilenet = timm.create_model(
            "mobilenetv3_small_050", pretrained=pretrained
        )
        self.mobilenet.classifier = nn.Linear(
            in_features=self.mobilenet.classifier.in_features,
            out_features=emb_size,
        )

    def forward(self, images):
        embeddings = self.mobilenet(images)
        return embeddings


if __name__ == "__main__":
    # model = SiameseNetwork()
    # model = EfficientNet()
    model = MobileNet()
    # device = set_device()
    device = "cpu"
    model.to(device)

    # Test network
    x = torch.rand(1, 3, 256, 128).to(device)
    embedding = model(x)
    print(embedding.shape)
    summary(model, x)
