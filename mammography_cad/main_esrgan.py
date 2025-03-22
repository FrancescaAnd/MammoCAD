import argparse
import os
from utils.split_dataset_esrgan import split_dataset_esrgan
from utils.esrgan_data import generate_pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'val', 'split','test'])
    parser.add_argument('--data', type=str, default="clahe", help='dataset to use (basic augmented or augmented plus clahe')
    args = parser.parse_args()


    if args.mode == "train":
        os.system("python train/esrgan_train.py")
    elif args.mode == "val":
        os.system("python eval/val_esrgan.py")
    elif args.mode == "split":
        if args.data == "clahe":
            generate_pairs("data/processed/clahePNG/", "data/esrgan_data/LR", "data/esrgan_data/HR", scale=4)
            split_dataset_esrgan()
        elif args.mode == "augmented":
            generate_pairs("data/processed/augmentedPNG/", "data/esrgan_data/LR", "data/esrgan_data/HR", scale=4)
            split_dataset_esrgan()
    elif args.mode == "test":
        os.system("python eval/test_esrgan.py")
