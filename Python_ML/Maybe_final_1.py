import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# --- Simple CNN Model Definition ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128), nn.ReLU(),
            nn.Linear(128, 2)  # Change 2 if your classes are different
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# --- Model Loading ---
model = SimpleCNN()
model.load_state_dict(torch.load("pcos_cnn_model.pth"))
model.eval()

# --- Image Processing ---
image_path = r"C:\Poornimaa\ASE\Python_ML\data\test\infected\img_0_571.jpg"  # Replace with your image path
class_names = ['cysts', 'no_cysts']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- Load Image ---
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# --- Prediction ---
output = model(input_tensor)
pred_class = output.argmax(1).item()

# --- Get Bounding Box Predictions (For this example, assuming output is in form: [x_center, y_center, width, height] --- change according to your model output)
# For simplicity, we'll assume there's only one box predicted and predict it manually
# You should change this part based on your model's actual output
# Example output: [class_id, x_center, y_center, width, height] (normalized to [0, 1])

pred_bboxes = []  # Assuming the model returns bounding boxes
# Example: [[0, 0.5, 0.5, 0.3, 0.3]] for a bounding box in the center
# Add code to parse model output accordingly

# --- Convert Predicted Coordinates to Pixel Values ---
img_w, img_h = image.size
for bbox in pred_bboxes:
    class_id, x_center, y_center, width, height = bbox
    x1 = int((x_center - width / 2) * img_w)
    y1 = int((y_center - height / 2) * img_h)
    x2 = int((x_center + width / 2) * img_w)
    y2 = int((y_center + height / 2) * img_h)

    # Draw Bounding Box on Image
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # Put the class label near the bounding box
    cv2.putText(image, class_names[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# --- Display the Image with Bounding Boxes ---
plt.imshow(image)
plt.title(f"Prediction: {class_names[pred_class]}")
plt.axis('off')
plt.show()


