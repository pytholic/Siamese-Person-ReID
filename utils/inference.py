import os
import sys

import torch
import torchvision.transforms as transforms
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

from config import logger
from siamese import SiameseNetwork

if __name__ == "__main__":
    model = SiameseNetwork()
    model.load_state_dict(
        torch.load("../models/best_model.pt", map_location=torch.device("mps"))
    )
    model.eval()

    logger.info("Model state loaded successfully...")

    image = Image.open(
        "/Users/3i-a1-2021-15/Developer/projects/pivo-tracking/similarity-metrics/test-images/1.jpeg"
    ).convert("RGB")

    transforms = transforms.Compose(
        [
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    img = transforms(image)
    img = torch.unsqueeze(img, 0)
    features = model(img)
    print(features)
