import argparse
from train.det_train import det_train
from utils.split_dataset import split_dataset
from utils.labels_det import labels_det
from eval.test_detection import det_evaluate

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 model for detection task")
    parser.add_argument("--mode", choices=["train", "test", "split"], required=True)
    parser.add_argument('--model', type=str, default="yolov8n.pt", help='YOLO model path (e.g., yolov8n.pt, yolov8m.pt)')
    parser.add_argument('--data', type=str, default="clahe", help='dataset to use (basic augmented or augmented plus clahe')
    parser.add_argument('--data_path', type=str, default="/home/francesca/Desktop/mammography_cad", help="Dataset path")
    parser.add_argument('--epochs', type=int, default=70, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')

    args = parser.parse_args()


    if args.mode == "train":
        # Call the training function with arguments
        det_train(
            model_path=args.model,
            epochs=args.epochs,
            batch=args.batch,
            data_path=args.data_path
        )
    elif args.mode == "test":
        det_evaluate("runs/detect/weights/best.pt", "data/processed/split_dataset_det/yolo_det.yaml", args.data_path)

    elif args.mode == "split":
        if args.data == "clahe":
            labels_det("data/processed/clahePNG/")
            split_dataset("data/processed/clahePNG", "data/labels/labels_det", "data/processed/split_dataset_det", "yolo_det.yaml")
        elif args.data == "augmented":
            labels_det("data/processed/augmentedPNG/")
            split_dataset("data/processed/augmentedPNG", "data/labels/labels_det","data/processed/split_dataset_det", "yolo_det.yaml")

if __name__ == "__main__":
    main()