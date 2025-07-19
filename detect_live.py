import cv2
from ultralytics import YOLO
# Load your trained model
model = YOLO("runs/detect/train3/weights/best.pt")
# Open webcam (or replace with your IP camera stream URL)
video_source = "rtsp://admin:admin123@192.168.0.191:554/ch1/main/av_stream"
 # Use 0 for webcam. For IP camera: video_source = "rtsp://username:password@your_camera_ip:port/stream_path"
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print("Could not open video source.")
    exit()
print("Detection started. Press ESC to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break
    results = model(frame)                # Run detection
    annotated = results[0].plot()         # Annotate results
    cv2.imshow("YOLOv8 Detection", annotated)  # Show window
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break
cap.release()
cv2.destroyAllWindows()
