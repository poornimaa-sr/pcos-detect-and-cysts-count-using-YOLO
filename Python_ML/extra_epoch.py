from ultralytics import YOLO

# Load a larger model for better accuracy (try 'yolov8m.pt' or 'yolov8l.pt')
model = YOLO('yolov8m.pt')  # You can also try yolov8l.pt or yolov8x.pt

# Train with more epochs, data augmentation, and image size

model.train(
    data='C:\Poornimaa\ASE\Python_ML\pcos--3\data.yaml',
    epochs=30,
    imgsz=640,
    batch=16,
    device='cpu', 
    augment=True,
    cos_lr=True,
    lr0=0.01,
    patience=20
)
metrics = model.val()
print(f"Accuracy (mAP@0.5): {metrics.box.map50:.4f}")

results = model("C:\Poornimaa\ASE\Python_ML\data\train\infected\img_0_245.jpg", conf=0.37)
results[0].show()
print("Detected classes:", results[0].boxes.cls.tolist())