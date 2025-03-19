import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from dataset.dataset_loader import MammogramDataset
from models.resnet import ResNetBinaryClassifier


def validate(json_path, img_dir, mask_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    val_dataset = MammogramDataset(json_path, img_dir, mask_dir, transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = ResNetBinaryClassifier().cuda()
    model.load_state_dict(torch.load("runs/resnet/resnet_classifier.pth"))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            outputs = model(imgs)
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # Precision, Recall, F1-Score
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

