import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json

class MammogramDataset(torch.utils.data.Dataset):
    ''' Create the dataset for classification stage'''
    def __init__(self, json_path, img_dir, mask_dir, transform=None):
        with open(json_path) as f:
            self.data = json.load(f)

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_keys = list(self.data.keys())

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        base_name = self.image_keys[idx]
        img_path = os.path.join(self.img_dir, base_name + ".png")
        mask_path = os.path.join(self.mask_dir, base_name + "_mask.png")

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        image = torch.cat((image, mask), dim=0)

        birads = str(self.data[base_name]["BIRADS"]).strip()
        birads_num = int(''.join(filter(str.isdigit, birads))) if birads else 2
        label = 0 if birads_num in [2, 3] else 1

        return image, torch.tensor(label, dtype=torch.long)
