import os
import cv2
import matplotlib.pyplot as plt
from collections import Counter

# --- Config ---
image_dir = r"C:\Poornimaa\ASE\Python_ML\pcos--3\train\images"
label_dir = r"C:\Poornimaa\ASE\Python_ML\pcos--3\train\labels"
class_names = ['cysts', 'no_cysts']
num_images = 3  # Number of images to show

# --- Get image files ---
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')][:num_images]

# --- Process each image ---
for image_name in image_files:
    image_path = os.path.join(image_dir, image_name)
    label_path = os.path.join(label_dir, image_name.replace('.jpg', '.txt'))

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    counts = Counter()

    # Draw bounding boxes + count annotations
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                class_id, x, y, bw, bh = map(float, parts)
                class_id = int(class_id)
                counts[class_id] += 1
                name = class_names[class_id]

                x1 = int((x - bw/2) * w)
                y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w)
                y2 = int((y + bh/2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Build title with counts
    title = image_name + " — " + ", ".join(
        f"{class_names[k]}: {v}" for k, v in sorted(counts.items())
    )

    # Display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()
    
