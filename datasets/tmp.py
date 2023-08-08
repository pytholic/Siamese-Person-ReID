from pathlib import Path

import torch
import torchvision.transforms as transforms
from dataloaders import *

if __name__ == "__main__":
    data_dir = Path(
        "/Users/3i-a1-2021-15/Developer/projects/datasets/DukeMTMC-reID"
    )
    transforms = transforms.Compose(
        [transforms.Resize((256, 128)), transforms.ToTensor()]
    )

    query_dataset = QueryDataset(data_dir=data_dir, transform=transforms)
    gallery_dataset = GalleryDataset(data_dir=data_dir, transform=transforms)

    print("Size of query set:", len(query_dataset))
    print("Size of gallery set:", len(gallery_dataset))
