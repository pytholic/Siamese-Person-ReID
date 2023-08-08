import lightning as L
from siamese import EfficientNet, MobileNet, MyNet, TripletLoss

fabric = L.Fabric(accelerator="cpu")
model = MobileNet()

state = fabric.load("best_model.ckpt")
model.load_state_dict(full_checkpoint["model"])

print(model)
