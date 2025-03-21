import os
import random
import shutil

def split_dataset_esrgan():
    # Paths to your dataset
    original_lr_dir = "data/esrgan_data/LR"
    original_hr_dir = "data/esrgan_data/HR"
    split_dir = "data/esrgan_data/split"  # Create a new directory for the split data

    # Split ratio
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # Create new directories for each split
    os.makedirs(f"{split_dir}/train/LR", exist_ok=True)
    os.makedirs(f"{split_dir}/train/HR", exist_ok=True)
    os.makedirs(f"{split_dir}/val/LR", exist_ok=True)
    os.makedirs(f"{split_dir}/val/HR", exist_ok=True)
    os.makedirs(f"{split_dir}/test/LR", exist_ok=True)
    os.makedirs(f"{split_dir}/test/HR", exist_ok=True)

    # Get all image file names
    lr_images = os.listdir(original_lr_dir)
    hr_images = os.listdir(original_hr_dir)

    # Ensure that both low-res and high-res images have the same filenames
    assert sorted(lr_images) == sorted(hr_images)

    # Split indices
    total_images = len(lr_images)
    indices = list(range(total_images))
    random.shuffle(indices)

    train_idx = int(total_images * train_ratio)
    val_idx = int(total_images * (train_ratio + val_ratio))

    train_images = [lr_images[i] for i in indices[:train_idx]]
    val_images = [lr_images[i] for i in indices[train_idx:val_idx]]
    test_images = [lr_images[i] for i in indices[val_idx:]]

    # Copy images into respective directories
    def copy_images(image_list, source_lr_dir, source_hr_dir, target_lr_dir, target_hr_dir):
        for image_name in image_list:
            shutil.copy(os.path.join(source_lr_dir, image_name), os.path.join(target_lr_dir, image_name))
            shutil.copy(os.path.join(source_hr_dir, image_name), os.path.join(target_hr_dir, image_name))

    copy_images(train_images, original_lr_dir, original_hr_dir, f"{split_dir}/train/LR", f"{split_dir}/train/HR")
    copy_images(val_images, original_lr_dir, original_hr_dir, f"{split_dir}/val/LR", f"{split_dir}/val/HR")
    copy_images(test_images, original_lr_dir, original_hr_dir, f"{split_dir}/test/LR", f"{split_dir}/test/HR")

    print(f"Dataset split: {train_idx} for training, {val_idx - train_idx} for validation, {total_images - val_idx} for testing")
