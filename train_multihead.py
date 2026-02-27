import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from Training.celeba_dataset import CelebaAttributes
from Training.model import PersonAttributeModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------- TRANSFORMS --------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# -------- DATASETS --------
train_dataset = CelebaAttributes(
    csv_file="train.csv",
    image_dir="Training/sorted_images",
    transform=transform
)

val_dataset = CelebaAttributes(
    csv_file="val.csv",
    image_dir="Training/sorted_images",
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# -------- MODEL --------
model = PersonAttributeModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

epochs = 5

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        for k in labels:
            labels[k] = labels[k].to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = (
            criterion(outputs["hair"], labels["hair"]) +
            criterion(outputs["beard"], labels["beard"]) +
            criterion(outputs["eye"], labels["eyewear"]) +
            criterion(outputs["gender"], labels["gender"]) +
            criterion(outputs["hat"], labels["hat"])
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"\nEpoch {epoch+1} Train Loss: {total_loss:.3f}")

    # -------- VALIDATION --------
    model.eval()

    correct = {
        "hair": 0,
        "beard": 0,
        "eye": 0,
        "gender": 0,
        "hat": 0
    }

    total = 0

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(device)
            for k in labels:
                labels[k] = labels[k].to(device)

            outputs = model(images)

            for key in correct:
                preds = torch.argmax(outputs[key], dim=1)
                correct[key] += (preds == labels[key]).sum().item()

            total += images.size(0)

    print("Validation Accuracy:")
    for key in correct:
        print(f"{key}: {correct[key] / total:.3f}")

# -------- SAVE MODEL --------
torch.save(model.state_dict(), "multihead_model.pth")
print("Model saved.")