import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.dataset_loader import MammogramDataset
from models.resnet import ResNetBinaryClassifier
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def eval(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Validation Metrics: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return accuracy, precision, recall, f1

def train(json_path, img_dir, mask_dir, val_json_path, epochs=20):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load datasets
    train_dataset = MammogramDataset(json_path, img_dir, mask_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    val_dataset = MammogramDataset(val_json_path, img_dir, mask_dir, transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize model, optimizer, and loss function
    model = ResNetBinaryClassifier().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss, correct = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {correct/len(train_loader.dataset)*100:.2f}%")

        # Evaluate the model on the validation set
        print(f"Evaluating after epoch {epoch+1}...")
        eval(model, val_loader, device='cuda')

    torch.save(model.state_dict(), "runs/resnet/resnet_classifier.pth")

