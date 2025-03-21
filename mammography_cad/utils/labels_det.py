import os
import json
import cv2
import numpy as np
from tqdm import tqdm

def labels_det(image_dir):
    json_path = "data/json/augmented.json"  # Path to JSON file
    #image_dir = "../data/processed/clahePNG/"  # Path to images
    yolo_labels_dir = "data/labels/labels_det"  # Output labels
    os.makedirs(yolo_labels_dir, exist_ok=True)
    os.path.exists(image_dir)


    # Assign numeric labels to ROI types
    class_mapping = {
        "Mass": 0,
        "Calcification": 1,
        "Distortion": 2,
        "Spiculated Region": 3
    }

    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # Process each image
    for img_id, img_data in tqdm(data.items()):
        img_filename = img_data["FileName"] + ".png"
        img_path = os.path.join(image_dir, img_filename)


        if not os.path.exists(img_path):
            print(f"Skipping {img_filename}, not found!")
            continue

        # Read image for dimensions
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {img_filename}, error loading image!")
            continue

        h, w, _ = img.shape
        label_filename = os.path.join(yolo_labels_dir, img_id + ".txt")

        # Process ROIs
        with open(label_filename, "w") as label_file:
            for roi in img_data["ROIs"]:
                if "Point_px" not in roi or len(roi["Point_px"]) < 2:
                    continue

                # Get bounding box (min/max x & y)
                points = np.array([eval(p) for p in roi["Point_px"]])
                x_min, y_min = points.min(axis=0)
                x_max, y_max = points.max(axis=0)

                # Normalize for YOLO format
                x_center = (x_min + x_max) / (2 * w)
                y_center = (y_min + y_max) / (2 * h)
                bbox_width = (x_max - x_min) / w
                bbox_height = (y_max - y_min) / h

                # Get class ID
                class_name = roi.get("Name", "Mass")  # Default to "Mass"
                class_id = class_mapping.get(class_name, 0)

                # YOLO format: class x_center y_center width height
                label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

    print("Conversion complete!")

