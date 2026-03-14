import cv2
import matplotlib.pyplot as plt
from collections import Counter

import tkinter as tk
from tkinter import filedialog

# Open file dialog to choose image
root = tk.Tk()
root.withdraw()  # Hide tkinter window
image_path = filedialog.askopenfilename(title="Select an Image")
img = cv2.imread(image_path)

# Paths
img_path  = r"C:\Poornimaa\ASE\Python_ML\pcos--3\train\images\img_0_567_jpg.rf.f171e6d144d4a35594a52755b59dce3b.jpg"

label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")

# Class names
class_names = ['cysts', 'no_cysts']
counts = Counter()

# Load image
img = cv2.imread(img_path)
h, w = img.shape[:2]

# Read labels and draw
with open(label_path, 'r') as f:
    for line in f:
        cls, x, y, bw, bh = map(float, line.strip().split())
        cls = int(cls)
        counts[cls] += 1
        name = class_names[cls]

        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Show image and count
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 8))
plt.imshow(img_rgb)
plt.title(f"cysts: {counts[0]}, no_cysts: {counts[1]}")
plt.axis("off")

num_cysts = counts[0]
# Determine condition
if 3<(num_cysts) <= 15:
    condition = "PCOD"
elif num_cysts <= 21:
    condition = "PCOS"
else:
    condition = "Severe PCOS"

# Add below image display
plt.figtext(0.5, -0.05, f"Cysts: {num_cysts} — Condition: {condition}", wrap=True, horizontalalignment='center', fontsize=12)

plt.show()



