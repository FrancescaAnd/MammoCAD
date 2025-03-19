import shutil
from sklearn.model_selection import train_test_split
import yaml
import os


def generate_yaml(output_dir, yaml_name, num_classes=4, class_names=None):
    '''Generates a yaml file for YOLO training.'''
    if class_names is None:
        class_names = ["Mass", "Calcification", "Distortion", "Spiculated Region"]

    dataset_yaml = {
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": num_classes,
        "names": class_names
    }

    yaml_path = os.path.join(output_dir, yaml_name)

    # Write the YAML file with the correct formatting
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, allow_unicode=True, indent=4)

    # Now post-process the 'names' key to ensure it is in the desired inline format
    with open(yaml_path, "r") as f:
        content = f.read()

    # Replace the names list format with the inline version
    content = content.replace("names:\n- Mass\n- Calcification\n- Distortion\n- Spiculated Region",
                              'names: ["Mass", "Calcification", "Distortion", "Spiculated Region"]')

    # Write the updated content back to the YAML file
    with open(yaml_path, "w") as f:
        f.write(content)

    print(f"{yaml_name} created at: {yaml_path}")



def split_dataset(image_dir, label_dir, output_dir, yaml_name, train_ratio=0.8, val_ratio=0.1):
    '''Splits the dataset into train, val, and test sets for yolo training'''
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    images = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    train_images, temp_images = train_test_split(images, train_size=train_ratio, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=val_ratio/(1-train_ratio), random_state=42)

    for split, split_images in zip(["train", "val", "test"], [train_images, val_images, test_images]):
        for img in split_images:
            shutil.copy(os.path.join(image_dir, img), os.path.join(output_dir, split, "images", img))
            label_file = img.replace(".png", ".txt")
            shutil.copy(os.path.join(label_dir, label_file), os.path.join(output_dir, split, "labels", label_file))
            
    # Generate dataset.yaml
    generate_yaml(output_dir, yaml_name)


#split_dataset("processed/clahePNG", "labels/det_labels", "processed/split_dataset_det", "yolo_det.yaml")
#split_dataset("processed/clahePNG", "labels/seg_labels", "processed/split_dataset_seg", "yolo_seg.yaml")

