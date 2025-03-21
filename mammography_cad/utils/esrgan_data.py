import os
from PIL import Image

'''Create LR-HR pairs from PNG images for ESRGAN training.
   Args:
        input_dir (str): Directory containing original high-resolution images.
        output_dir_lr (str): Directory to save low-resolution images.
        output_dir_hr (str): Directory to save high-resolution images.
        scale (int): Downsampling factor to create low-resolution images.
'''
def generate_pairs(input_dir, output_dir_lr, output_dir_hr, scale=4):
    os.makedirs(output_dir_lr, exist_ok=True)
    os.makedirs(output_dir_hr, exist_ok=True)

    print("Please wait for LR-HR pairs generation...")

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert("L")

            # Save HR version
            hr_save_path = os.path.join(output_dir_hr, filename)
            img.save(hr_save_path)

            # Create LR by downsampling
            lr_img = img.resize((img.width // scale, img.height // scale), Image.BICUBIC)
            lr_img = lr_img.resize((img.width, img.height), Image.BICUBIC)  # Upsample back to original size

            # Save LR
            lr_save_path = os.path.join(output_dir_lr, filename)
            lr_img.save(lr_save_path)

    print("LR-HR pairs created!")

# Example usage
