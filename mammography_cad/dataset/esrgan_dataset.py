import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ESRGANDataset(Dataset):
    '''Dataset class for loading low-resolution and high-resolution image pairs'''
    def __init__(self, lr_dir, hr_dir, lr_transform=None, hr_transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform
        self.filenames = sorted([f for f in os.listdir(lr_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        lr_image = Image.open(os.path.join(self.lr_dir, self.filenames[idx])).convert("L")
        hr_image = Image.open(os.path.join(self.hr_dir, self.filenames[idx])).convert("L")

        if self.lr_transform:
            lr_image = self.lr_transform(lr_image)
        if self.hr_transform:
            hr_image = self.hr_transform(hr_image)

        return lr_image, hr_image
