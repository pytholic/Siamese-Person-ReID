from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as transforms
from dataloaders import *
from torch.utils.data import DataLoader

from utils import plot_triplet

if __name__ == "__main__":
    data_dir = Path(
        "/Users/3i-a1-2021-15/Developer/projects/datasets/DukeMTMC-reID"
    )

    transforms = transforms.Compose(
        [transforms.Resize((256, 128)), transforms.ToTensor()]
    )

    dataset = SiameseDukemtmcreidDataset(
        data_dir=data_dir, transform=transforms
    )
    trainset, validset = torch.utils.data.random_split(dataset, [0.95, 0.05])

    print(f"Size of trainset : {len(trainset)}")
    print(f"Size of validset : {len(validset)}")

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    validloader = DataLoader(validset, batch_size=32)

    print(f"No. of batches in trainloader : {len(trainloader)}")
    print(f"No. of batches in validloader : {len(validloader)}")

    for A, P, N in trainloader:
        triplet = (A[0], P[0], N[0])
        break

    plot_triplet(triplet)
