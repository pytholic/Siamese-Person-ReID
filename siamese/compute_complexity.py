import torch
from network import SiameseNetwork
from ptflops import get_model_complexity_info

model = SiameseNetwork()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

macs, params = get_model_complexity_info(
    model,
    (3, 256, 128),
    as_strings=True,
    print_per_layer_stat=True,
    verbose=True,
    flops_units="GMac",
    param_units="M",
)
print("{:<30}  {:<8}".format("Computational complexity: ", macs))
print("{:<30}  {:<8}".format("Number of parameters: ", params))
