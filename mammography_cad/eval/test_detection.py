from ultralytics import YOLO
import os

def det_evaluate(
        model_path="runs/detect/weights/best.pt",
        data_yaml="data/processed/split_dataset_det/yolo_det.yaml",
        data_path="/home/francesca/Desktop/mammography_cad",
        device="cuda",
):
    # Load YOLO model
    print(f"Loading YOLO model for evaluation: {model_path}")
    model = YOLO(model_path)

    # Full path to data YAML
    data_yaml_path = os.path.join(data_path, data_yaml)
    print("Starting evaluation...")

    # Evaluate
    results = model.val(data=data_yaml_path, device=device)

    # Extract metrics dictionary
    metrics = results.results_dict

    # Print main metrics
    precision = metrics['metrics/precision(B)']
    recall = metrics['metrics/recall(B)']
    mAP50 = metrics['metrics/mAP50(B)']
    mAP50_95 = metrics['metrics/mAP50-95(B)']

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"mAP@50: {mAP50:.4f}")
    print(f"mAP@50-95: {mAP50_95:.4f}")

    # F1 Score Calculation
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    print(f"F1 Score: {f1:.4f}")

    # NOTE: Accuracy in object detection is tricky and not directly computed by YOLOv8.
    # You could approximate accuracy by treating True Positives vs total predictions
    # But YOLOv8 does not give labels/preds arrays directly like classification.

    print(f"Evaluation complete. Results saved at: {results.save_dir}")
    return metrics



