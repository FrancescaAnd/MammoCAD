import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset.dataset_loader import MammogramDataset
from models.resnet import ResNetBinaryClassifier

def evaluate(json_path, img_dir, mask_dir, model_path):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load test dataset
    test_dataset = MammogramDataset(json_path, img_dir, mask_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load model
    model = ResNetBinaryClassifier().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Lists to store true and predicted labels
    all_labels = []
    all_preds = []

    # Disable gradient computation
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            outputs = model(imgs)
            preds = outputs.argmax(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(" Test Set Evaluation ")
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


