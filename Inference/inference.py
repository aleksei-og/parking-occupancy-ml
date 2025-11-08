#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
from ultralytics import YOLO
from models_arch import ResidualBlock, BetterCNNWithResiduals, SmallCNN
import cv2


# In[ ]:


def load_model(path, model_type):
    if model_type == 0:
        model = torch.load(path)
        model.eval()
    else:
        model = YOLO(path)
    return model


# In[ ]:


mean = torch.tensor([0.0070, 0.0069, 0.0065])
std = torch.tensor([0.0036, 0.0035, 0.0035])

val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


# In[ ]:


def predict_parking_spots(model, frame, areas, threshold=0.95):
    preds = []
    inputs = []

    for (x1, y1, x2, y2) in areas:
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            preds.append(0)
            continue
        img = Image.fromarray(cropped[:, :, ::-1])
        tensor = val_transform(img)
        inputs.append(tensor)

    if not inputs:
        return [(0, 0, 255)] * len(areas)

    batch = torch.stack(inputs)
    with torch.no_grad():
        outputs = model(batch)
        probs = torch.sigmoid(outputs).squeeze().numpy()
    return [(0, 255, 0) if p > threshold else (0, 0, 255) for p in probs]


# In[ ]:


def load_areas_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    areas = []
    for ann in data["annotations"]:
        if ann.get("category_id") == 1:
            continue
        x, y, w, h = ann["bbox"]
        areas.append((int(x), int(y), int(x + w), int(y + h)))
    return areas


# In[ ]:


def create_annotations(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Не удалось прочитать последний кадр видео")

    model_path = "best.pt" 
    model = YOLO(model_path)
    results = model(frame)
    
    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    annotations = []
    for idx, (x1, y1, x2, y2) in enumerate(xyxy, start=1):
        w = x2 - x1
        h = y2 - y1
        annotations.append({
            "id": idx,
            "image_id": 1,
            "category_id": int(cls_ids[idx - 1]),
            "bbox": [float(x1), float(y1), float(w), float(h)]
        })
    
    categories = [
        {"id": 0, "name": "emptylot"},
        {"id": 1, "name": "moving car"},
        {"id": 2, "name": "nonemptylot"}
    ]

    output = {
        "annotations": annotations,
        "categories": categories
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Аннотации сохранены в {output_path}")

