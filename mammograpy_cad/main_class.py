import argparse
from train.class_train import train
from eval.val_class import validate
from eval.test_class import evaluate
from utils.split_dataset_class import split_dataset_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modified ResNet classification model")
    parser.add_argument("--mode", choices=["train", "val","test", "split"], required=True)
    parser.add_argument("--epochs", type=int, default=80)


    args = parser.parse_args()

    if args.mode == "train":
        train('data/json/class/train.json', 'data/processed/clahePNG', 'data/mass_masks', 'data/json/class/val.json', epochs=args.epochs)
    elif args.mode == "val":
        validate('data/json/class/val.json', 'data/processed/clahePNG', 'data/mass_masks')
    elif args.mode == "split":
        split_dataset_class('data/json/augmented.json', 'data/json/class')
    elif args.mode == "test":
        evaluate('data/json/class/test.json', 'data/processed/clahePNG', 'data/mass_masks', 'runs/resnet/resnet_classifier.pth')
