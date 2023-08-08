import random
from pathlib import Path

import pandas as pd

data_dir = Path("/Users/3i-a1-2021-15/Developer/projects/datasets/MSMT17_V1")

triplets = set()

with open(data_dir / "train.txt", "r") as f:
    image_list = [line.strip() for line in f]

for row in image_list:
    anchor_path = row.split(" ")[0].strip()
    anchor_id = anchor_path.split("/")[0]

    positive_paths = [
        path
        for path in image_list
        if path.split("/")[0] == anchor_id and path != anchor_path
    ]
    positive_path = (
        random.choice(positive_paths) if positive_paths else anchor_path
    ).split(" ")[0]

    negative_paths = [
        path for path in image_list if path.split("/")[0] != anchor_id
    ]
    negative_path = random.choice(negative_paths).split(" ")[0]

    triplets.add((anchor_path, positive_path, negative_path))

df = pd.DataFrame(triplets, columns=["Anchor", "Positive", "Negative"])
csv_file = "train.csv"
df.to_csv(csv_file, index=True)
