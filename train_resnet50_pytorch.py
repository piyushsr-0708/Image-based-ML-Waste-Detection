import os
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, random_split

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ===============================
# Logging Setup
# ===============================
log_filename = f"training_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

def log_print(*args):
    """Print to console AND write to log file."""
    text = " ".join(str(a) for a in args)
    print(text)
    logging.info(text)


# ===============================
# Configurations
# ===============================
dataset_dir = r"C:\Users\HP\Desktop\ML Waste Classifier\dataset\combined_dataset"
batch_size = 50
num_epochs = 15
img_size = 96

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_print(f"Using device: {device}")


# ===============================
# Transforms
# ===============================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(img_size + 32),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ===============================
# Dataset Loading
# ===============================
included_classes = ['Hazardous', 'Organic', 'Recyclable']
full_dataset = datasets.ImageFolder(dataset_dir, transform=train_transform)

idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}
filtered_samples = [s for s in full_dataset.samples if idx_to_class[s[1]] in included_classes]

class_to_idx_new = {cls: i for i, cls in enumerate(included_classes)}

full_dataset.samples = [(path, class_to_idx_new[idx_to_class[label]]) for path, label in filtered_samples]
full_dataset.targets = [s[1] for s in full_dataset.samples]
full_dataset.class_to_idx = class_to_idx_new

num_classes = len(included_classes)

log_print("Dataset distribution:")
for cls in included_classes:
    c = full_dataset.targets.count(class_to_idx_new[cls])
    log_print(f"{cls}: {c}")
log_print(f"Total images: {len(full_dataset)}")


# ===============================
# Train/Validation Split
# ===============================
num_train = int(0.8 * len(full_dataset))
num_val = len(full_dataset) - num_train

train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val])
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# ===============================
# Class Weights (Handling Imbalance)
# ===============================
train_labels = [full_dataset.samples[i][1] for i in train_dataset.indices]
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

log_print("Class Weights:", class_weights.cpu().numpy())


# ===============================
# Model Setup (ResNet50 TL)
# ===============================
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

for name, param in model.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False  # freeze early layers

model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)


# ===============================
# Training Loop
# ===============================
train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_acc = 0.0

log_print("=== Training Started ===")
total_start_time = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()
    model.train()

    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validation
    model.eval()
    val_running_loss, val_correct, val_total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)

            val_running_loss += loss.item() * inputs.size(0)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = val_running_loss / val_total
    val_acc = val_correct / val_total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    epoch_duration = time.time() - epoch_start

    log_print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
        f"Epoch Time: {epoch_duration:.2f} sec"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model_resnet50.pth")
        log_print(">>> Best model updated.")


# ===============================
# Total Training Time
# ===============================
total_time = time.time() - total_start_time
log_print(f"=== Total Training Time: {total_time/60:.2f} minutes ({total_time:.2f} sec) ===")


# ===============================
# Confusion Matrix + Classification Report
# ===============================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=included_classes,
            yticklabels=included_classes,
            cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

report = classification_report(all_labels, all_preds, target_names=included_classes)
with open("classification_report.txt", "w") as f:
    f.write(report)

log_print("Classification Report:")
log_print(report)


# ===============================
# Accuracy & Loss Curves
# ===============================
epochs_range = range(1, num_epochs + 1)

plt.plot(epochs_range, train_accs, label="Train Accuracy")
plt.plot(epochs_range, val_accs, label="Validation Accuracy")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.savefig("accuracy_curve.png", dpi=300)
plt.close()

plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, val_losses, label="Validation Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.savefig("loss_curve.png", dpi=300)
plt.close()

log_print("=== Training Completed Successfully ===")