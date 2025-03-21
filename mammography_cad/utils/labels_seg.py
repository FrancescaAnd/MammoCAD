import os
import json
import cv2
import numpy as np

def labels_seg(image_dir):
    # Paths
    json_path = "data/json/augmented.json"  # JSON file path
    #image_dir = "../data/processed/clahePNG/"  # Path to images
    output_label_dir = "data/labels/labels_seg"  # Output labels

    # Create directories
    os.makedirs(output_label_dir, exist_ok=True)

    # Class Mapping (Assign numeric labels to ROI types)
    class_mapping = {
        "Mass": 0,
        "Calcification": 1,
        "Distortion": 2,
        "Spiculated Region": 3
    }

    # Load JSON data
    with open(json_path, "r") as f:
        data = json.load(f)

    # Process images
    for img_id, img_data in data.items():
        img_filename = img_data["FileName"] + ".png"
        img_path = os.path.join(image_dir, img_filename)

        # Check if image exists
        if not os.path.exists(img_path):
            print(f"Skipping {img_filename}, not found!")
            continue

        # Load image to get dimensions
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {img_filename}, error loading image!")
            continue
        height, width, _ = img.shape  # Get image dimensions

        # Prepare label file
        yolo_label_path = os.path.join(output_label_dir, img_filename.replace(".png", ".txt"))
        with open(yolo_label_path, "w") as label_file:
            # Process each ROI
            for roi in img_data["ROIs"]:
                if "Point_px" not in roi or len(roi["Point_px"]) < 3:  # Minimum 3 points for a polygon
                    continue

                # Extract segmentation contour points
                points = np.array([eval(p) for p in roi["Point_px"]], dtype=np.float32)

                # Normalize points to YOLO format (relative to image size)
                normalized_points = [(x / width, y / height) for x, y in points]
                normalized_points_flat = " ".join([f"{x:.6f} {y:.6f}" for x, y in normalized_points])

                # Get class ID
                class_name = roi.get("Name", "Mass")  # Default to "Mass"
                class_id = class_mapping.get(class_name, 0)

                # Write segmentation annotation in YOLO format
                label_file.write(f"{class_id} {normalized_points_flat}\n")

    print(" YOLO segmentation dataset preparation complete!")
