import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys
import time

# Suppress FFmpeg errors
sys.stderr = open(os.devnull, 'w')

# Load optimized YOLO model (ensure you use your ONNX model for better performance)
model = YOLO("yash.onnx", task='detect')  # Use the ONNX model

# RTSP stream setup
video_path = 'rtsp://admin:mauli1234@192.168.1.64:554/Streaming/Channels/101'
video_path_out = './ip_cam_output.mp4'

# Set environment variables for better RTSP stream handling
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Detection flag setup
flag_file_path = "detect_flag.txt"

# Function to reconnect to RTSP stream
def connect_stream():
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_FPS, 24)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    return cap

# Initial connection attempt
cap = connect_stream()

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

frame_count = 0
skip_frames = 12
retry_count = 0
max_retries = 5
retry_delay = 1

INTERESTING_CLASSES = [1 ,2, 3, 5, 7]
class_names = {1: "Bicycle",2: "Car", 3: "Bike", 5: "Bus", 7: "Truck"}

ret, frame = cap.read()
if not ret:
    print("Error: Failed to read first frame.")
    exit()

H, W, _ = frame.shape
fps = 5
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), 5, (W, H))

while True:
    ret, frame = cap.read()

    if not ret:
        print("Warning: Lost connection. Retrying...")
        retry_count += 1
        if retry_count <= max_retries:
            print(f"Attempting to reconnect... ({retry_count}/{max_retries})")
            time.sleep(retry_delay)
            retry_delay *= 2
            cap.release()
            cap = connect_stream()
            continue
        else:
            print("Error: Unable to reconnect to the RTSP stream.")
            break

    retry_count = 0
    retry_delay = 1

    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    results = model.predict(frame, imgsz=640, conf=0.35, iou=0.4)

    detection_found = False

    for res in results:
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        classes = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()

        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[i]
            conf = confs[i]
            class_id = classes[i]

            if class_id in INTERESTING_CLASSES and conf > 0.35:
                label = class_names.get(class_id, "Unknown")
                detection_found = True

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    # Write detection flag (1 if detection found, else 0)
    with open(flag_file_path, "w") as f:
        f.write("1" if detection_found else "0")

    out.write(frame)
    display_frame = cv2.resize(frame, (640, 480))
    cv2.imshow('Processed Video', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
