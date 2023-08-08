from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class SiameseMSMT17Dataset(Dataset):
    data_dir: Path = None
    transform: transforms.Compose = None

    def __post_init__(self):
        self.train_dir = self.data_dir / "train"
        self.csv_train = self.data_dir / "train.csv"
        self.df = pd.read_csv(self.csv_train)

    def __getitem__(self, index: int) -> tuple:
        row = self.df.iloc[index]

        anchor = Image.open(self.train_dir / row.Anchor).convert("RGB")
        positive = Image.open(self.train_dir / row.Positive).convert("RGB")
        negative = Image.open(self.train_dir / row.Negative).convert("RGB")

        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

    def __len__(self):
        return len(self.df)
