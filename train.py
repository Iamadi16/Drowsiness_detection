import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms , models
from torch.utils.data import DataLoader 

transform = transforms.Compose([
    transforms.Resize((224, 224)), #MobileNet accepts this size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load dataset
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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Classes:", train_dataset.class_to_idx)
print("Train Images:", len(train_dataset))
print("Validation Images:", len(val_dataset))
print("Test Images:", len(test_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Model
model = models.mobilenet_v2(weights="DEFAULT") #Default weights

model.classifier[1] = nn.Linear( #classifier->which class the image belongs to
    model.last_channel,
    2
)

model = model.to(device)

# Loss - optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

# train
epochs = 10
best_val_ac = 0.0

for epoch in range(epochs):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    # validation
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {running_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

    #save best one
    if val_accuracy > best_val_ac:
        best_val_ac = val_accuracy
        torch.save(
            model.state_dict(),
            "drowsiness_model.pth"
        )
        print("best model saved.")

# test
model.load_state_dict(
    torch.load(
        "drowsiness_model.pth",
        map_location= device
    )
)

model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total

print(f"Test Accuracy: {test_accuracy:.2f}%")

print("Training Finished!")