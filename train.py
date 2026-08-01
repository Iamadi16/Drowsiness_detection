import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms , models
from torch.utils.data import DataLoader

# تغییر اندازه تصاویر
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# خواندن دیتاست
train_dataset = datasets.ImageFolder(
    "dataset/train_data",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    "dataset/validation_data",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    "dataset/test_data",
    transform=transform
)

# ساخت DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

print("Classes:", train_dataset.classes)
print("Train Images:", len(train_dataset))
print("Validation Images:", len(val_dataset))
print("Test Images:", len(test_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = models.mobilenet_v2(weights="DEFAULT")

model.classifier[1] = nn.Linear(
    model.last_channel,
    2
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
epochs = 3

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}  Loss: {running_loss:.4f}")

torch.save(model.state_dict(), "drowsiness_model.pth")

print("Model Saved!")