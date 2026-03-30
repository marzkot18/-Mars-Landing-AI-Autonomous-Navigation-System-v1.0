# ==========================================
# TRAINING SCRIPT WITH RESNET18 (PRETRAINED)
# ==========================================

import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from app import MarsDataset
from torchvision import models
import torch.nn as nn

# ------------------------------------------
# 1. SET RANDOM SEED (for reproducibility)
# ------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed()

# ------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------
data_path = r"C:\Users\marzk\Downloads\SafeLanding\SafeLanding\safelanding\Auburn_1"
dataset = MarsDataset(data_path)

print(f"Total images: {len(dataset)}")

# ------------------------------------------
# 3. TRAIN / VALIDATION SPLIT
# ------------------------------------------
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ------------------------------------------
# 4. MODEL SETUP (PRETRAINED RESNET18)
# ------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Use pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 8)

# Freeze early layers for faster convergence
for param in model.conv1.parameters(): param.requires_grad = False
for param in model.layer1.parameters(): param.requires_grad = False

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()

# ------------------------------------------
# 5. TRAINING LOOP
# ------------------------------------------
epochs = 50
best_val_loss = float("inf")

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ---------------- VALIDATION ----------------
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, labels)
            val_loss += loss.item()

            predicted = torch.argmax(preds, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total

    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {accuracy:.2f}%")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "model.pth")
        print("✅ Best model saved!")

# ------------------------------------------
# 6. DONE
# ------------------------------------------
print("\n🎉 Training Complete!")
print("Best model saved as model.pth")