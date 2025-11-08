#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from inference import predict_parking_spots
from ultralytics import YOLO


# In[11]:


def process_video(config, model, areas):
    cap = cv2.VideoCapture(config["video_input"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(config["video_output"], fourcc, fps, (width, height))
    
    frame_counter = 0
    alpha = config["box_alpha"]
    model_type = config["model_type"]
    area_colors = [(0, 0, 255) for _ in areas]
    class_colors = {
        "moving_car": (255, 255, 255),
        "emptylot": (0, 255, 0),
        "nonemptylot": (0, 0, 255)
    }
    results = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        overlay = frame.copy()
        empty_count = 0
        total_spots = 0
        
        if model_type == 0:
            if frame_counter % config["frame_step"] == 0:
                area_colors = predict_parking_spots(model, frame, areas, config["confidence_threshold"])
            for (x1, y1, x2, y2), color in zip(areas, area_colors):
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            total_spots = len(areas)
            empty_count = sum(1 for color in area_colors if color == (0, 255, 0))
            
        else:
            if frame_counter % config["frame_step"] == 0:
                results = model.predict(frame, conf=0.5, iou=0.4, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    color = class_colors.get(label, (255, 255, 255))
                    
                    if label != "moving car":
                        total_spots += 1
                        if label == "emptylot":
                            empty_count += 1
                        
                    if model_type == 1:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        if label == "moving car":
                            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:
                        if label in ["emptylot", "nonemptylot"]:
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                        elif (label == "moving car") and config["frame_step"] < 5:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if model_type == 2: 
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
        status_text = f"Free: {empty_count} / {total_spots}"
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out.write(frame)
        cv2.imshow('Parking Status', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_counter += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




