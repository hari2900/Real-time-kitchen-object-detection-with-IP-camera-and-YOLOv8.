# 🍴 Kitchen Utensils Object Detection using YOLOv8

This project performs real-time object detection on kitchen utensils such as forks, knives, spoons, plates, mugs, glasses, and tongs using the YOLOv8 deep learning model.

## 📦 Dataset

We used the [Kitchen Utensils Dataset](https://www.kaggle.com/datasets/raunakgola/kitchen-utensils) from Kaggle. The dataset includes images categorized into 7 classes:
- Fork
- Knife
- Plate
- Spoon
- Mug
- Glass
- Tongs

## 🧠 Model Training

- We used [Ultralytics YOLOv8]
- The training was performed on the preprocessed dataset with a `data.yaml` file specifying paths and class names.
- A pretrained model `yolov8n.pt` was fine-tuned for the utensils dataset.
- Best weights were saved at: `runs/detect/train3/weights/best.pt`

## 🎥 Real-time Detection

This project supports both **webcams** and **IP cameras** for live object detection.

### 📷 IP Camera Setup
To use an IP camera, replace the `video_source` in `detect_live.py` with your camera’s RTSP stream URL:
```python
video_source = "rtsp://username:password@your_camera_ip:port/stream_path"
```
Example:
```python
video_source = "rtsp://admin:admin123@192.168.0.191:554/ch1/main/av_stream"
```
Make sure the IP camera is connected to the same network and accessible.

Run the detection script:
```bash
python detect_live.py
```

## 📁 Files in this Repository

| File                        | Description                                        |
|----------------------------|----------------------------------------------------|
| `detect_live.py`           | Runs live object detection using trained model     |
| `data.yaml`                | YOLO configuration file for class names and paths  |
| `yolov8n.pt`               | Pretrained model used for training/fine-tuning     |

## ✅ Classes

The following utensils are detectable:
1. Fork
2. Knife
3. Plate
4. Spoon
5. Mug
6. Glass
7. Tongs

## 📌 Requirements

- Python 3.8+
- OpenCV
- scikit-learn
- Ultralytics YOLOv8:
  ```bash
  pip install ultralytics
  ```

---

## 📊 Model Performance

The YOLOv8 model was trained for 50 epochs and achieved high accuracy across all kitchen utensil classes.

![Training Results](training_results.png)

### Sample mAP Scores (on validation set):
- **mAP@0.5**: 0.924  
- **mAP@0.5:0.95**: 0.813  

Per class performance:

| Class  | mAP@0.5 | mAP@0.5:0.95 |
|--------|---------|--------------|
| Fork   | 0.973   | 0.851        |
| Knife  | 0.995   | 0.883        |
| Plate  | 0.995   | 0.915        |
| Spoon  | 0.616   | 0.526        |
| Mug    | 0.995   | 0.889        |
| Glass  | 0.897   | 0.732        |
| Tongs  | 0.995   | 0.892        |

> Training was completed in approximately **2.65 hours** on a CPU (Intel i5-1135G7).
