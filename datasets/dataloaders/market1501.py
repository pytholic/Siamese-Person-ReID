import random
from dataclasses import dataclass
from pathlib import Path

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class SiameseMarket1501Dataset(Dataset):
    data_dir: Path
    transform: transforms.Compose = None

    @staticmethod
    def get_id(path: Path) -> str:
        return str(path.parts[-1]).split("_")[0]

    def __post_init__(self):
        self.train_dir = self.data_dir / "bounding_box_train"
        self.image_list = list(self.train_dir.glob("*.jpg"))

    def __getitem__(self, index: int) -> tuple:
        anchor_path = self.image_list[index]
        anchor_id = self.get_id(anchor_path)

        positive_paths = [
            path
            for path in self.image_list
            if self.get_id(path) == anchor_id and path != anchor_path
        ]

        positive_path = (
            random.choice(positive_paths) if positive_paths else anchor_path
        )

        negative_paths = [
            path for path in self.image_list if self.get_id(path) != anchor_id
        ]
        negative_path = random.choice(negative_paths)

        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")

        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

    def __len__(self):
        return len(self.image_list)
