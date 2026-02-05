"""
Train YOLO on xView Dataset for Satellite Container Detection
60 classes including Container Ship, Shipping Container, Maritime Vessel
"""
from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolo11n.pt")

# Train on xView dataset with absolute path
print("Starting xView training...")
print("Container-related classes: Container Ship, Shipping Container, Container Crane")
print("Plus: Maritime Vessel, Cargo Ship, Cargo Truck, Barge, etc.")

results = model.train(
    data="/home/misango/codechest/Algorithmic_Trading_and_HFT_Research/Satellite_Container_Analysis/datasets/xview/xView.yaml",
    epochs=50,  # Start with 50 epochs
    imgsz=640,
    batch=8,
    device=0,  # Use GPU
    project="runs/detect",
    name="xview_train",
    patience=10,
    save=True,
    plots=True,
    exist_ok=True
)

print("\n✓ Training complete!")
print(f"Best model saved to: runs/detect/xview_train/weights/best.pt")
