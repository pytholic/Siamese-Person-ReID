import glob
import os
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from numpy.linalg import norm
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

from config import logger
from siamese import SiameseNetwork


def cosine_similarity(arr1, arr2):
    dist = np.dot(arr1, arr2) / (norm(arr1) * norm(arr2))
    return dist


def euclidean_distance(arr1, arr2):
    dist = np.linalg.norm(arr1 - arr2)
    return dist


if __name__ == "__main__":
    model = SiameseNetwork()
    model.load_state_dict(
        torch.load("../models/best_model.pt", map_location="cpu")
    )
    model.eval()

    logger.info("Model state loaded successfully...")

    images = sorted(
        glob.glob(
            "/Users/3i-a1-2021-15/Developer/projects/pivo-tracking/similarity-metrics/test-images/*"
        )
    )

    transforms = transforms.Compose(
        [
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    features = []

    for image in images:
        img = Image.open(image).convert("RGB")
        img = transforms(img)
        img = torch.unsqueeze(img, 0)
        res = model(img)

        features.append(res.detach().numpy().squeeze(axis=0))

    # features = np.array(features)
    for i in range(0, 6):
        dist_diff = cosine_similarity(features[4], features[i])
        print(f"Distance: {dist_diff}")
