import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# Load YOLOv8 model once
model = YOLO(r"C:\Poornimaa\ASE\Python_ML\runs\detect\train3\weights\best.pt")  # Update path

def select_and_predict():
    # Open file dialog
    image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not image_path:
        return
    
    # Run prediction
    results = model(image_path, conf=0.089)
    
    # Count cysts (class 0)
    boxes = results[0].boxes
    cyst_count = sum(1 for cls_id in boxes.cls if int(cls_id) == 0)
    
    # Display results
    results[0].show()
    results[0].save()  # Saved to runs/detect/predict/
    
    print(f"The Number of Cysts Detected: {cyst_count}")

# Tkinter UI
root = tk.Tk()
root.title("YOLOv8 Cyst Detection")
root.geometry("300x100")

btn = tk.Button(root, text="Select Image", command=select_and_predict)
btn.pack(pady=30)

root.mainloop()


from ultralytics import YOLO

# Load model
model = YOLO(r"C:\Poornimaa\ASE\Python_ML\runs\detect\train3\weights\best.pt")  # Replace with your model path

# Evaluate and get accuracy
metrics = model.val(data=r"C:\Poornimaa\ASE\Python_ML\pcos--3\data.yaml")  # Replace with your dataset config
accuracy = metrics.box.map50  # mAP@0.5 is considered as accuracy

print(f"Accuracy (mAP@0.5): {accuracy:.4f}")
    