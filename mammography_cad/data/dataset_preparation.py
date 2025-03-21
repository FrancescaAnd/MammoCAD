import os
import numpy as np
import cv2
from glob import glob
import pydicom
import random
import shutil
from matplotlib import pyplot as plt
import json_preparation
# INbreast Paths
RAW_DATA_PATH1 = "raw/AllDICOMs/"
PROCESSED_DATA_PATH1 = "processed/"
CONVERTED_DATA_PATH1 = f"{PROCESSED_DATA_PATH1}/AllPNG" #converted from DICOM to PNG format
AUG_DATA_PATH = f"{PROCESSED_DATA_PATH1}/augmentedPNG" # Augmented dataset path

# Directories
os.makedirs(CONVERTED_DATA_PATH1, exist_ok=True)
os.makedirs(AUG_DATA_PATH, exist_ok=True)

def convert_format():
    ''' Converts .dcm images (inside datasets) to PNG format '''
    dicom_files = glob(f"{RAW_DATA_PATH1}/*.dcm", recursive=True)
    print(f"Images in .dcm format: {len(dicom_files)}")

    for dicom_path in dicom_files:
        try:
            dicom = pydicom.dcmread(dicom_path)
            img_array = dicom.pixel_array  # Extract pixel data

            # Normalize pixel values (convert to 8-bit range)
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255.0
            img_array = img_array.astype(np.uint8)
            
            # Extract filename (before first underscore) to use as new PNG filename
            filename = os.path.splitext(os.path.basename(dicom_path))[0]  # Get file name without extension
            name = filename.split('_')[0]  # Get the part before the first underscore
            output_path = os.path.join(CONVERTED_DATA_PATH1, f"{name}.png")

            cv2.imwrite(output_path, img_array)

        except Exception as e:
            print(f"Error processing {dicom_path}: {e}")

    print("Conversion of DICOM to PNG complete!")

def augment_images():
    ''' Performs data augmentation on PNG images and saves them.
    Augmentations: contrast adjustment, noise addition.
    '''

    img_files = glob(f"{CONVERTED_DATA_PATH1}/*.png")
    print(f"Applying augmentation to {len(img_files)} images...")

    for img_path in img_files:
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read grayscale image
            filename = os.path.basename(img_path).split('.')[0]  # Get filename without extension

            aug_imgs = []


            # Copy original image to augmented folder
            shutil.copy(img_path, f"{AUG_DATA_PATH}/{filename}.png")


            # Contrast Adjustment
            alpha = random.uniform(0.8, 1.2)  # Brightness factor
            beta = random.randint(-20, 20)  # Contrast factor
            contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            contrast = np.clip(contrast, 0, 255).astype(np.uint8)  # Clip values to ensure valid range
            aug_imgs.append((contrast, f"{filename}_contrast.png"))

            # Gaussian Noise Addition
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            noisy = cv2.add(img, noise)
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)  # Clip values to ensure valid range
            aug_imgs.append((noisy, f"{filename}_noise.png"))

            # Save augmented images
            for aug_img, aug_filename in aug_imgs:
                cv2.imwrite(f"{AUG_DATA_PATH}/{aug_filename}", aug_img)


        except Exception as e:
            print(f"Error augmenting {img_path}: {e}")

    # Count the number of images in AUG_DATA_PATH after augmentation
    aug_image_count = len(glob(f"{AUG_DATA_PATH}/*.png"))
    print(f"Total number of images in {AUG_DATA_PATH}: {aug_image_count}")

    print("Data augmentation complete!")


def apply_clahe(clip_limit=2.0, tile_grid_size=(8, 8)):
    '''Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to
    all PNG images in the input directory.'''
    input_dir = 'processed/augmentedPNG/'
    output_dir = 'processed/clahePNG/'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Please wait for CLAHE application...")


    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Apply CLAHE to each PNG image
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale

            # Apply CLAHE
            img_clahe = clahe.apply(image)

            # Save the enhanced image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img_clahe)
    print(f"Images enhanced by CLAHE!")


if __name__ == "__main__":
    print("Preprocessing of dataset images in progress, please wait...")
    convert_format() # First, converting .dcm to .png
    augment_images() # Then, augmenting the dataset
    apply_clahe() # Enhance images
    json_preparation.main()  #prepare the json dictionary with images info
    print("Preprocessing complete!")

