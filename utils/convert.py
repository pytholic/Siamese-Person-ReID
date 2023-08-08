import os
import sys

import onnx
import torch
from torch.autograd import Variable
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

from config import logger
from siamese import MobileNet, MyNet

# model = MobileNet()
model = MyNet()
model.load_state_dict(
    torch.load("/home/app/code/best_model.pt", map_location=torch.device("cpu"))
)
# Define a new model without the final classification layer (if needed)
class ModifiedModel(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel, self).__init__()
        # Copy all layers from the original model except the last one (classification layer)
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        return self.features(x)

# Create the modified model
# model = ModifiedModel(model_orig)
model.eval()

logger.info("Model state loaded successfully...")

input_name = ["input"]
output_name = ["output"]
input = torch.randn(1, 3, 256, 128)
torch.onnx.export(
    model,
    input,
    "./model.onnx",
    input_names=input_name,
    output_names=output_name,
    verbose=True,
    export_params=True,
)

logger.info("Model converted successfully...")

onnx_model = onnx.load("./model.onnx")
onnx.checker.check_model(onnx_model)

logger.info("Onnx model loaded successfully...")
