import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.esrgan import Generator
from dataset.esrgan_dataset import ESRGANDataset
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchvision import transforms

# Hyperparameters
batch_size = 2
img_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "runs/esrgan/generator.pth"
val_lr_dir = "data/esrgan_data/split/val/LR"
val_hr_dir = "data/esrgan_data/split/val/HR"

# Dataset and DataLoader for validation
# Transforms
lr_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Keep LR small
    transforms.ToTensor(),
])

hr_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # HR stays large
    transforms.ToTensor(),
])

val_dataset = ESRGANDataset(
    lr_dir=val_lr_dir,
    hr_dir=val_hr_dir,
    lr_transform=lr_transform,
    hr_transform=hr_transform
)


val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load the trained generator model
generator = Generator().to(device)
generator.load_state_dict(torch.load(checkpoint_path))
generator.eval()

# Initialize metrics
psnr_metric = PeakSignalNoiseRatio().to(device)
ssim_metric = StructuralSimilarityIndexMeasure().to(device)

total_val_psnr = 0
total_val_ssim = 0

# Evaluate the generator on the validation set
with torch.no_grad():
    for i, (val_lr_imgs, val_hr_imgs) in enumerate(val_dataloader):
        val_lr_imgs, val_hr_imgs = val_lr_imgs.to(device), val_hr_imgs.to(device)
        fake_val_imgs = generator(val_lr_imgs)

        # Save images for monitoring
        if i % 10 == 0:  # Save every 10th batch
            save_image(fake_val_imgs, f"runs/esrgan/val_{i}.png")

        # Calculate PSNR and SSIM for evaluation
        psnr = psnr_metric(fake_val_imgs, val_hr_imgs)
        ssim = ssim_metric(fake_val_imgs, val_hr_imgs)
        total_val_psnr += psnr.item()
        total_val_ssim += ssim.item()

    avg_psnr = total_val_psnr / len(val_dataloader)
    avg_ssim = total_val_ssim / len(val_dataloader)
    print(f"Validation PSNR:{avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")



