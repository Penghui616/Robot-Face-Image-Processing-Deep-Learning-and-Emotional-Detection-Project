import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Select device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model structure (5 classes)
class ModifiedVGG16(nn.Module):
    def __init__(self, num_classes=5, fine_tune=True):
        super(ModifiedVGG16, self).__init__()
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # Freeze all convolutional layers
        for param in self.vgg16.features.parameters():
            param.requires_grad = False
        
        # Replace the classifier
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(25088, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
        # Optionally fine-tune the last few convolutional layers
        if fine_tune:
            for param in self.vgg16.features[-8:].parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.vgg16(x)

# Image preprocessing
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Define custom dataset (5 classes)
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

# Load test dataset
test_dir = "E:/5class/test"
test_dataset = EmotionDataset(test_dir, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load trained model
model = ModifiedVGG16(num_classes=5).to(device)
model.load_state_dict(torch.load("E:/best_model_5class.pth", map_location=device))
model.eval()

# Evaluate model
test_correct = 0
test_total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * test_correct / test_total
print(f"\nTest Set Accuracy: {test_accuracy:.2f}%\n")

print("Classification report:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

# Visualize confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=test_dataset.classes,
            yticklabels=test_dataset.classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
