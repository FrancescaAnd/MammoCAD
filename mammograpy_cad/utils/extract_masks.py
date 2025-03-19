'''generate binary masks from your dataset and crop mass regions for classification'''

import json
import os
import cv2
import numpy as np
from PIL import Image

# Paths
json_path = "data/json/augmented.json"  # Path to dataset.json
image_dir = "data/processed/clahePNG/"  # Mammogram images path
mask_dir = "data/mass_masks/"  # Output directory for binary masks
os.makedirs(mask_dir, exist_ok=True)

# Load dataset JSON
with open(json_path, "r") as f:
    dataset = json.load(f)

# Process each image
for img_id, img_data in dataset.items():
    img_filename = img_data["FileName"] + ".png"
    img_path = os.path.join(image_dir, img_filename)

    # Check if image exists
    if not os.path.exists(img_path):
        print(f"Skipping {img_filename}, not found!")
        continue

    # Load image to get dimensions
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Skipping {img_filename}, error loading image!")
        continue
    height, width = img.shape

    # Create an empty mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Process each ROI
    for roi in img_data["ROIs"]:
        if "Point_px" not in roi or len(roi["Point_px"]) < 3:
            continue  # Skip invalid ROIs

        # Convert contour points to numpy array
        points = np.array([eval(p) for p in roi["Point_px"]], dtype=np.int32)

        # Fill the contour in the mask
        cv2.fillPoly(mask, [points], 255)  # Fill with white (255)

    # Save mask
    mask_filename = os.path.join(mask_dir, img_data["FileName"] + "_mask.png")
    cv2.imwrite(mask_filename, mask)

print("Binary mask generation complete! Masks saved in:", mask_dir)
