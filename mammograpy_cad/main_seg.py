import argparse
from train.seg_train import seg_train
from utils.split_dataset import split_dataset
from utils.labels_seg import labels_seg
from eval.test_segmentation import seg_evaluate


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 model for segmentation task")
    parser.add_argument("--mode", choices=["train", "test", "split"], required=True)
    parser.add_argument('--model', type=str, default="yolov8n_seg.pt", help='YOLO model path (e.g., yolov8n_seg.pt, yolov8m_seg.pt)')
    parser.add_argument('--data', type=str, default="clahe", help='dataset to use (basic augmented or augmented plus clahe')
    parser.add_argument('--data_path', type=str, default="/home/francesca/Desktop/mammography_cad", help="Dataset path")
    parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')

    args = parser.parse_args()


    if args.mode == "train":
        # Call the training function with arguments
        seg_train(
            model_path=args.model,
            epochs=args.epochs,
            batch=args.batch,
            data_path=args.data_path
        )
    elif args.mode == "test":
        seg_evaluate("runs/segment/weights/best.pt", "data/processed/split_dataset_seg/yolo_seg.yaml", args.data_path)

    elif args.mode == "split":
        if args.data == "clahe":
            labels_seg("data/processed/clahePNG/")
            split_dataset("data/processed/clahePNG", "data/labels/labels_seg", "data/processed/split_dataset_seg", "yolo_seg.yaml")
        elif args.data == "augmented":
            labels_seg("data/processed/augmentedPNG/")
            split_dataset("data/processed/augmentedPNG", "data/labels/labels_seg","data/processed/split_dataset_seg", "yolo_seg.yaml")

if __name__ == "__main__":
    main()