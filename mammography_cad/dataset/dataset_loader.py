import os
import json
import torch
from PIL import Image
from torchvision import transforms
import numpy as np

class MammogramDataset(torch.utils.data.Dataset):
    ''' Dataset for per-mass (ROI-level) classification '''
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

        # Load and convert to grayscale
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Convert mask to numpy array to find ROI
        mask_np = np.array(mask)
        if mask_np.max() > 0:  # Check if mask contains any ROI
            coords = np.column_stack(np.where(mask_np > 0))
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            # Crop both image and mask to bounding box
            image = image.crop((x_min, y_min, x_max, y_max))
            mask = mask.crop((x_min, y_min, x_max, y_max))
        else:
            # If no ROI in mask, return full image
            pass

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Concatenate along channel dimension
        image = torch.cat((image, mask), dim=0)

        # Get label (0: benign, 1: malignant)
        birads = str(self.data[base_name]["BIRADS"]).strip()
        birads_num = int(''.join(filter(str.isdigit, birads))) if birads else 2
        label = 0 if birads_num in [2, 3] else 1

        return image, torch.tensor(label, dtype=torch.long)
