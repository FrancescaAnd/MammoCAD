from ultralytics import YOLO
import os

def seg_train(
    model_path="yolov8n_seg.pt",
    data_yaml="mammography_cad/data/processed/split_dataset_seg/yolo_seg.yaml",
    data_path="/home/francesca/Desktop/",
    epochs=70,
    batch=4,
    imgsz=1024,
    patience=20,
    lr0=0.0005,
    optimizer="AdamW",
    weight_decay=0.0005,
    amp=True,
    device="cuda",
    save_dir="runs_seg/",
):

    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    # Modify the data argument to use the environment variable DATA_PATH
    data_yaml_path = os.path.join(data_path, data_yaml)  # Construct the full path to the YAML file

    print("Starting training...")
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=patience,
        lr0=lr0,
        optimizer=optimizer,
        weight_decay=weight_decay,
        amp=amp,
        device=device,
        save=True,
        save_dir=save_dir,
    )
    print("Training completed!")