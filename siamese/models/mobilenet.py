import timm
import torch
import torch.nn as nn
from torchsummary import summary


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
    model = MobileNet()

    # Test network
    x = torch.rand(1, 3, 256, 128)
    embedding = model(x)
    print(embedding.shape)
    summary(model, x)
