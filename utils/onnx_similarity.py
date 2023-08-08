import glob

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision.transforms as transforms
from numpy.linalg import norm
from PIL import Image


def cosine_similarity(arr1, arr2):
    dist = np.dot(arr1, arr2) / (norm(arr1) * norm(arr2))
    return dist


if __name__ == "__main__":
    onnx_model = onnx.load("../models/model.onnx")
    onnx.checker.check_model(onnx_model)

    ort_sess = ort.InferenceSession(
        "../models/model.onnx", disabled_optimizers=["EliminateDropout"]
    )
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

        res = ort_sess.run(None, {"input": img.numpy()})
        res = np.array(res).squeeze(axis=0).squeeze(axis=0)

        features.append(res)

    features = np.array(features)
    for i in range(0, 6):
        dist_diff = cosine_similarity(features[0], features[i])
        print(f"Distance: {dist_diff}")
