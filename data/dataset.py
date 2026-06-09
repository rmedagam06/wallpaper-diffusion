from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import json


class WallpaperDataset(Dataset):
    def __init__(self, metadata_path, size=512):
        with open(metadata_path) as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        self.transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["image"]).convert("RGB")
        return self.transform(img), row["caption"]
