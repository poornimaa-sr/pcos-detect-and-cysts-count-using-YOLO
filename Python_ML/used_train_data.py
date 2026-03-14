import os
import shutil

# --- Configuration ---
base_source = r"C:\Poornimaa\ASE\Python_ML\pcos--3"
base_target = r"C:\Poornimaa\ASE\Python_ML\pcos_classification"
splits = ['train', 'valid']
class_names = ['cysts', 'no_cysts']

# --- Create target folders ---
for split in splits:
    for cls in class_names:
        os.makedirs(os.path.join(base_target, split, cls), exist_ok=True)

# --- Process each split ---
for split in splits:
    img_dir = os.path.join(base_source, split, "images")
    lbl_dir = os.path.join(base_source, split, "labels")

    for lbl_file in os.listdir(lbl_dir):
        if not lbl_file.endswith(".txt"):
            continue
        lbl_path = os.path.join(lbl_dir, lbl_file)
        img_name = lbl_file.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        with open(lbl_path) as f:
            line = f.readline().strip()
            if line:
                class_id = int(line.split()[0])
                class_folder = class_names[class_id]
                target_path = os.path.join(base_target, split, class_folder, img_name)
                shutil.copy(img_path, target_path)

#----------CHECK ACCURACY----------

from ultralytics import YOLO

# Load model
model = YOLO(r"C:\Poornimaa\ASE\Python_ML\runs\detect\train3\weights\best.pt")  # Replace with your model path

# Evaluate and get accuracy
metrics = model.val(data=r"C:\Poornimaa\ASE\Python_ML\pcos--3\data.yaml")  # Replace with your dataset config
accuracy = metrics.box.map50  # mAP@0.5 is considered as accuracy

print(f"Accuracy (mAP@0.5): {accuracy:.4f}")

