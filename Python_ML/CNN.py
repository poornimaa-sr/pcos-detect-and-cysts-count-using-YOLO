import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# --- Config ---
data_dir = r"C:\Poornimaa\ASE\Python_ML\pcos_classification"
batch_size = 16
epochs = 10
num_classes = 2  # cysts, no_cysts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- Load Dataset ---
train_data = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
valid_data = datasets.ImageFolder(root=f"{data_dir}/valid", transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size)

# --- Simple CNN ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = SimpleCNN().to(device)

# --- Loss and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop ---
for epoch in range(epochs):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    acc = correct / len(train_data)
    print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Train Acc={acc:.4f}")

# --- Save Model ---
torch.save(model.state_dict(), "pcos_cnn_model.pth")


#-----------------LOAD THE TRAINED MODEL----------

model = SimpleCNN() 
model.load_state_dict(torch.load("pcos_cnn_model.pth"))
model.eval().to(device)


from PIL import Image
import torchvision as transforms

# --- Config ---
image_path = r"C:\Poornimaa\ASE\Python_ML\data\test\infected\img_0_571.jpg"  # test image
class_names = ['cysts', 'no_cysts']

# --- Transform ---

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



# --- Load & Predict ---
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]
output = model(input_tensor)
pred_class = output.argmax(1).item()

# --- Result ---
print(f"Prediction: {class_names[pred_class]}")
print("Since cysts are predcited, there are chances of PCOS detected.")

# --- Evaluate on Validation Data ---
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

val_acc = correct / total
print(f"Validation Accuracy: {val_acc:.4f}")

