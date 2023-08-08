import time

import onnx
import onnxruntime as ort
import torch
import torchvision.transforms as transforms
from PIL import Image

onnx_model = onnx.load("../models/model.onnx")
onnx.checker.check_model(onnx_model)

ort_sess = ort.InferenceSession(
    "../models/model.onnx", disabled_optimizers=["EliminateDropout"]
)

image = Image.open(
    "/Users/3i-a1-2021-15/Developer/projects/pivo-tracking/similarity-metrics/test-images/1.jpeg"
).convert("RGB")

transforms = transforms.Compose(
    [
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

img = transforms(image)
img = torch.unsqueeze(img, 0)
print(img.shape)

start = time.perf_counter()
outputs = ort_sess.run(None, {"input": img.numpy()})
end = time.perf_counter()

print(outputs)
print(f"Inference time = {end-start}")
