import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Select GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define modified VGG16 model (5 classes)
class ModifiedVGG16(nn.Module):
    def __init__(self, num_classes=5, fine_tune=True):
        super(ModifiedVGG16, self).__init__()
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # Freeze convolutional layers
        for param in self.vgg16.features.parameters():
            param.requires_grad = False
        
        # Replace classifier with custom layers
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(25088, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
        # Optionally fine-tune last few layers
        if fine_tune:
            for param in self.vgg16.features[-8:].parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.vgg16(x)

# Data augmentation for training set
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(8),
    transforms.RandomAffine(degrees=8, translate=(0.03, 0.03)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Preprocessing for validation/test set
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Custom dataset (5 emotion classes)
class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = ['angry', 'happy', 'neutral', 'sad', 'surprised']

        for label_idx, class_name in enumerate(self.classes):
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.exists(class_folder):
                continue
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.long)

# Dataset paths
train_dir = "E:/5class/train"

# Build full dataset and split into training and validation sets
full_train_dataset = EmotionDataset(train_dir, transform=transform_train)
train_indices, val_indices = train_test_split(range(len(full_train_dataset)), test_size=0.2, random_state=42)
train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_train_dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model, loss function, optimizer, and scheduler
model = ModifiedVGG16(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training configuration
epochs = 10
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(100 * correct / total)

    # Validation phase
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(100 * val_correct / val_total)

    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, "
          f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.2f}%")

    scheduler.step(val_losses[-1])

    # Save the best model
    if val_accuracies[-1] == max(val_accuracies):
        torch.save(model.state_dict(), "E:/best_model_5class.pth")
        print(f"New best model found! (Val Acc: {val_accuracies[-1]:.2f}%)")

print("Training complete. Model saved as 'E:/best_model_5class.pth'")

# Plot training and validation loss and accuracy curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", linestyle="--", marker="o")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss", linestyle="-", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy", linestyle="--", marker="o")
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Val Accuracy", linestyle="-", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Curve")
plt.legend()
plt.show()
