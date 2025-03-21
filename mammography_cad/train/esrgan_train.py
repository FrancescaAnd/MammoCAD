import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.esrgan import Generator, Discriminator
from dataset.esrgan_dataset import ESRGANDataset
from utils.esrgan_utils import ContentLoss, adversarial_loss, VGGFeatureExtractor
import os
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch import amp

# Mixed precision scaler
scaler = amp.GradScaler()

torch.cuda.empty_cache()

cudnn.benchmark = False
cudnn.deterministic = True
cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Paths for dataset splits
train_lr_dir = "data/esrgan_data/split/train/LR"
train_hr_dir = "data/esrgan_data/split/train/HR"

# Hyperparameters
batch_size = 2
epochs = 20
img_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# Training Dataset
train_dataset = ESRGANDataset(lr_dir=train_lr_dir, hr_dir=train_hr_dir, hr_transform=transform, lr_transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Models
generator = Generator().to(device)
discriminator = Discriminator().to(device)
feature_extractor = VGGFeatureExtractor().to(device)
content_loss = ContentLoss(feature_extractor).to(device)

# Optimizers
optimizer_G = optim.AdamW(generator.parameters(), lr=1e-4)
optimizer_D = optim.AdamW(discriminator.parameters(), lr=1e-4)

# Gradient accumulation steps
accumulation_steps = 2

# Training Loop
for epoch in range(epochs):
    print("Start training...")
    generator.train()
    discriminator.train()
    total_g_loss = 0
    total_d_loss = 0

    with tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{epochs}') as pbar:
        for i, (lr_imgs, hr_imgs) in pbar:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            # Discriminator Update
            optimizer_D.zero_grad()
            with amp.autocast('cuda'):
                real_preds = discriminator(hr_imgs)
                fake_imgs = generator(lr_imgs)
                fake_preds = discriminator(fake_imgs.detach())

                d_loss_real = adversarial_loss(real_preds, True)
                d_loss_fake = adversarial_loss(fake_preds, False)
                d_loss = (d_loss_real + d_loss_fake) * 0.5

            # Accumulate gradients for discriminator
            scaler.scale(d_loss).backward()
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer_D)
                scaler.update()
                optimizer_D.zero_grad()

            # Generator Update
            optimizer_G.zero_grad()
            with amp.autocast('cuda'):
                fake_preds = discriminator(fake_imgs)
                adv_loss = adversarial_loss(fake_preds, True)
                cont_loss = content_loss(fake_imgs, hr_imgs)
                g_loss = cont_loss + 1e-3 * adv_loss

            scaler.scale(g_loss).backward()
            scaler.step(optimizer_G)
            scaler.update()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

            torch.cuda.empty_cache()

            # Update progress bar with loss values
            pbar.set_postfix(G_Loss=total_g_loss / (i + 1), D_Loss=total_d_loss / (i + 1))

    print(f"Epoch [{epoch+1}/{epochs}] G_Loss: {total_g_loss:.4f} | D_Loss: {total_d_loss:.4f}")

    # Save model every epoch
    os.makedirs("runs/esrgan", exist_ok=True)
    torch.save(generator.state_dict(), "runs/esrgan/generator.pth")
    torch.save(discriminator.state_dict(), "runs/esrgan/discriminator.pth")

print("Training complete. Models saved.")
