import timm
import torch
import torch.nn as nn
from torchsummary import summary


class EfficientNet(nn.Module):
    def __init__(self, emb_size=512):
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


if __name__ == "__main__":
    model = EfficientNet()
    # Test network
    x = torch.rand(1, 3, 256, 128)
    embedding = model(x)
    print(embedding.shape)
    summary(model, x)
