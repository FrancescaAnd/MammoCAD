import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.esrgan import Generator
from dataset.esrgan_dataset import ESRGANDataset
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchvision import transforms

def evaluate_esrgan(test_lr_dir, test_hr_dir):
    # Hyperparameters
    batch_size = 2
    img_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "runs/esrgan/generator.pth"

    lr_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Keep LR small
        transforms.ToTensor(),
    ])

    hr_transform = transforms.Compose([
        transforms.Resize((512, 512)),  # HR stays large
        transforms.ToTensor(),
    ])

    test_dataset = ESRGANDataset(
        lr_dir=test_lr_dir,
        hr_dir=test_hr_dir,
        lr_transform=lr_transform,
        hr_transform=hr_transform
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # Load the trained generator model
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(checkpoint_path))
    generator.eval()

    # Metrics used
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)
    psnr_metric = PeakSignalNoiseRatio().to(device)

    total_test_ssim = 0
    total_test_psnr = 0

    # Evaluate the generator on the test set
    with torch.no_grad():
        for i, (test_lr_imgs, test_hr_imgs) in enumerate(test_dataloader):
            test_lr_imgs, test_hr_imgs = test_lr_imgs.to(device), test_hr_imgs.to(device)
            fake_test_imgs = generator(test_lr_imgs)

            # Save images for monitoring
            if i % 10 == 0:
                save_image(fake_test_imgs, f"runs/esrgan/test_{i}.png")

            # Calculate SSIM and PSNR for evaluation
            ssim = ssim_metric(fake_test_imgs, test_hr_imgs)
            psnr = psnr_metric(fake_test_imgs, test_hr_imgs)

            total_test_ssim += ssim.item()
            total_test_psnr += psnr.item()

        # Average metrics
        avg_test_ssim = total_test_ssim / len(test_dataloader)
        avg_test_psnr = total_test_psnr / len(test_dataloader)

    print(f"Test SSIM: {avg_test_ssim:.4f}")
    print(f"Test PSNR: {avg_test_psnr:.4f}")

    return avg_test_ssim, avg_test_psnr

test_lr_dir = "data/esrgan_data/split/test/LR"
test_hr_dir = "data/esrgan_data/split/test/HR"
evaluate_esrgan(test_lr_dir, test_hr_dir)
