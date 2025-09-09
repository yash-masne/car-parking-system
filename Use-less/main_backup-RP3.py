import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys
import time
import RPi.GPIO as GPIO
from RPLCD.i2c import CharLCD
import signal # Added for graceful exit

# --- Constants ---
LED_PIN = 23 # Status LED pin
# IMPORTANT: Reverting INFERENCE_IMG_SIZE to 640 because your ONNX model expects 640x640 inputs.
# For true optimization, you should re-export/retrain your 'yash.onnx' model
# to accept smaller input sizes (e.g., 320 or 416).
INFERENCE_IMG_SIZE = 640 # Must match the input size your yash.onnx model expects
MODEL_CONF_THRESHOLD = 0.35 # Minimum confidence for object detection
MODEL_IOU_THRESHOLD = 0.4   # IoU threshold for Non-Maximum Suppression
# Keeping FRAME_SKIP_INTERVAL high to compensate for slow inference
FRAME_SKIP_INTERVAL = 50 # Process every 51st frame (skips 50 frames)
OUTPUT_VIDEO_FPS = 5        # FPS for the output video file
VIDEO_CLEANUP_INTERVAL_HOURS = 1 # Interval in hours to clean up the output video file

# --- GPIO SETUP ---
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)  # Default OFF

# --- LCD SETUP ---
# Store LCD dimensions as variables immediately after initialization
LCD_COLS = 20
LCD_ROWS = 4
lcd = CharLCD('PCF8574', 0x27, cols=LCD_COLS, rows=LCD_ROWS)
lcd.clear()
lcd.write_string("Starting in ") # Initial message as per your code

# --- Signal Handler for Graceful Exit ---
def signal_handler(sig, frame):
    """Handles Ctrl+C (SIGINT) for a clean script exit."""
    print("\nExiting gracefully...")
    lcd.clear()
    lcd.write_string("Shutting Down...")
    time.sleep(1) # Give LCD time to display
    # Ensure cleanup runs on Ctrl+C exit
    if 'cap' in globals() and cap is not None and cap.isOpened():
        cap.release()
    if 'out' in globals() and out is not None and out.isOpened():
        out.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    lcd.clear()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# --- Startup Countdown (Camera Warmup) ---
for i in range(30, 0, -1):
    lcd.home()
    lcd.write_string(f"Starting in {i}s")
    if i==10:
        time.sleep(1)
        lcd.clear()
    if i!=10:
        time.sleep(1)
lcd.clear()
lcd.write_string("Initializing")
lcd.clear() # Redundant after lcd.clear(), but harmless if you prefer it

# --- Main Application Loop ---
try:
    # --- Load YOLO Model ---
    try:
        model = YOLO("yash.onnx", task='detect')
        GPIO.output(LED_PIN, GPIO.HIGH)  # Success: LED ON
        lcd.write_string("Model Loaded OK")
    except Exception as e:
        # If model loading fails, we immediately raise an exception
        # to jump to the outer critical error handling
        raise RuntimeError(f"Model loading failed: {e}")

    # --- RTSP Setup ---
    video_path = 'rtsp://admin:mauli1234@192.168.1.64:554/Streaming/Channels/101'
    video_output_path = './ip_cam_output.mp4'

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    flag_file_path = "detect_flag.txt"

    # Connect stream function
    def connect_stream():
        """Attempts to connect to the RTSP video stream."""
        cap_local = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        # It's better to let the model resize to 640x640 for inference,
        # so we don't try to force a different input resolution on the camera here.
        # However, if your camera supports exactly 640x360 or 640x480, setting it
        # can reduce internal OpenCV overhead. For now, removing the
        # forced lower resolution setting to avoid conflicts with model input.
        # cap_local.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap_local.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) # Or 480 depending on camera aspect ratio
        cap_local.set(cv2.CAP_PROP_FPS, 24) # Desired FPS for stream
        return cap_local

    # Global cap and out for access in signal handler and cleanup
    cap = None
    out = None

    cap = connect_stream()

    if not cap.isOpened():
        raise RuntimeError("Initial RTSP stream connection failed.")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read first frame from stream.")

    frame_count = 0
    retry_count = 0
    max_retries = 5
    retry_delay = 1

    INTERESTING_CLASSES = [2, 3, 5, 7, 10]
    class_names = {2: "Car", 3: "Bike", 5: "Bus", 7: "Truck", 10: "Tractor"}

    # Use actual frame dimensions for video writer (these will be from the RTSP stream)
    H, W, _ = frame.shape
    out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), OUTPUT_VIDEO_FPS, (W, H))
    last_cleanup_time = time.time()

    # --- Continuous Processing Loop ---
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Warning: Lost connection. Retrying...")
            retry_count += 1
            lcd.clear()
            lcd.write_string("Reconnecting...")
            if retry_count <= max_retries:
                print(f"Attempting to reconnect... ({retry_count}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2 # Exponential backoff for retry delay
                if cap is not None:
                    cap.release()
                cap = connect_stream()
                continue
            else:
                raise RuntimeError("Unable to reconnect to RTSP stream after multiple attempts.")

        retry_count = 0 # Reset retry count on successful frame read
        retry_delay = 1 # Reset retry delay

        frame_count += 1
        if frame_count % FRAME_SKIP_INTERVAL != 0: # Use constant for clarity
            continue

        # Using INFERENCE_IMG_SIZE (which is now 640) for prediction.
        # The YOLO library will handle the necessary resizing to 640x640 if the input frame
        # from the camera is not already that exact size.
        results = model.predict(frame, imgsz=INFERENCE_IMG_SIZE, conf=MODEL_CONF_THRESHOLD, iou=MODEL_IOU_THRESHOLD)
        detection_found = False
        vehicle_count = {2: 0, 3: 0, 5: 0, 7: 0, 10: 0}
        # ADDED: Dictionary to store the highest confidence for each detected class in the current frame
        vehicle_confidence = {2: 0.0, 3: 0.0, 5: 0.0, 7: 0.0, 10: 0.0}

        for res in results:
            boxes = res.boxes.xyxy.cpu().numpy().astype(int)
            classes = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()

            for i in range(len(boxes)):
                xmin, ymin, xmax, ymax = boxes[i]
                conf = confs[i]
                class_id = classes[i]

                # Ensure confidence matches MODEL_CONF_THRESHOLD for consistency
                if class_id in INTERESTING_CLASSES and conf > MODEL_CONF_THRESHOLD:
                    label = class_names.get(class_id, "Unknown")
                    vehicle_count[class_id] += 1
                    detection_found = True
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
                    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                    # ADDED: Store the highest confidence for this class
                    if conf > vehicle_confidence[class_id]:
                        vehicle_confidence[class_id] = conf

        # Write detection flag
        with open(flag_file_path, "w") as f:
            f.write("1" if detection_found else "0")

        out.write(frame)

        # Update LCD with vehicle info
        lcd.home()
        lcd.write_string("  Mauli Water Park                             Ready        Waiting for Vehicles")
        line = 0
        for cid in INTERESTING_CLASSES:
            if vehicle_count[cid] > 0: # Only display if count is greater than 0
                lcd.clear()
                lcd.cursor_pos = (line, 0)
                # ADDED: Format confidence as percentage
                confidence_percent = int(vehicle_confidence[cid] * 100)
                # MODIFIED: Include confidence percentage in the LCD string
                # Ensure the string fits within LCD_COLS
                display_string = f"{class_names[cid]} : {vehicle_count[cid]}  -> {confidence_percent}%"
                if len(display_string) > LCD_COLS:
                    display_string = display_string[:LCD_COLS] # Truncate if too long
                lcd.write_string(display_string)
                line += 1
                if line >= LCD_ROWS: # Use the constant variable
                    break
                        
        # Clear the last line if not used by vehicle counts
        if line == LCD_ROWS: # Use the constant variable
            lcd.cursor_pos = (LCD_ROWS - 1, 0) # Use the constant variable
            lcd.write_string(" " * LCD_COLS) # Use the constant variable


        # Periodically clean up the output video file
        if time.time() - last_cleanup_time >= (VIDEO_CLEANUP_INTERVAL_HOURS * 3600): # Use constant
            print(f"Cleaning up old video file: {video_output_path}")
            if out is not None:
                out.release() # Release current writer
            if os.path.exists(video_output_path):
                os.remove(video_output_path) # Delete the file
            out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), OUTPUT_VIDEO_FPS, (W, H)) # Create new writer
            last_cleanup_time = time.time() # Reset timer

# --- Critical Error Handling Loop ---
except Exception as e:
    print(f"CRITICAL SYSTEM FAILURE: {e}")
    # This loop will run forever, displaying the error and preventing exit
    while True:
        try:
            GPIO.output(LED_PIN, GPIO.LOW) # Turn off LED
            lcd.clear()
            lcd.cursor_pos = (0, 0)
            lcd.write_string("CRITICAL ERROR")
            lcd.cursor_pos = (1, 0)
            lcd.write_string("SYSTEM HALTED")
            lcd.cursor_pos = (2, 0)
            # Correctly get the error type name and truncate if necessary
            error_type_name = type(e).__name__
            # Use the constant variable LCD_COLS
            if len(error_type_name) > LCD_COLS:
                error_type_name = error_type_name[:LCD_COLS - 3] + "..."
            lcd.write_string(f"[{error_type_name}]")
            # You can add the specific error message too if it fits
            # For example:
            # error_msg = str(e)
            # if len(error_msg) > LCD_COLS: # Use the constant variable
            #     error_msg = error_msg[:LCD_COLS - 3] + "..."
            # lcd.cursor_pos = (3,0)
            # lcd.write_string(error_msg)

        except Exception as lcd_err:
            # If LCD itself fails, print to console as a last resort
            print(f"ERROR: Could not update LCD during critical error: {lcd_err}")
        time.sleep(5) # Display for a few seconds, then try to refresh (or just wait)
        # No resource cleanup, no sys.exit(). The process stays alive here.
