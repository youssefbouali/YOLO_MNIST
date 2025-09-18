
#!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="nuHD8IjmNxjGCJ2tP9cl")
project = rf.workspace("mnist-bvalq").project("mnist-icrul")
version = project.version(8)
dataset = version.download("yolov8")

#!pip install ultralytics


from ultralytics import YOLO
import torch

# --- Check if GPU is available ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Load YOLOv8 nano model ---
model = YOLO("yolov8n.pt")

# --- Train on MNIST with GPU ---
model.train(
    data="MNIST-8/data.yaml",
    epochs=20,
    imgsz=400,
    device=device,  # <-- specify GPU
    task="detect"
)
