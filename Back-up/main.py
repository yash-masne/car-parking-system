import cv2
import os
from ultralytics import YOLO
from gpiozero import LED
from RPLCD.i2c import CharLCD
import signal
import time
import sys
import subprocess
import threading

# Set OpenCV FFMPEG options for RTSP globally
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp;max_delay;1000000"

# Constants
LED_PIN = 23
INFERENCE_IMG_SIZE = 640
MODEL_CONF_THRESHOLD = 0.35
MODEL_IOU_THRESHOLD = 0.4
FRAME_SKIP_INTERVAL = 20 # Frame skip for individual cameras

LCD_COLS, LCD_ROWS = 20, 4

INTERESTING_CLASSES = [2, 3, 5, 7, 10]
class_names = {2: "Car", 3: "Bike", 5: "Bus", 7: "Truck", 10: "Tractor"}

# Global variables for inter-thread communication and LCD management
global_detections = {
    "cam1": {"count": {cls: 0 for cls in INTERESTING_CLASSES}, "confidence": {cls: 0.0 for cls in INTERESTING_CLASSES}, "found": False},
    "cam2": {"count": {cls: 0 for cls in INTERESTING_CLASSES}, "confidence": {cls: 0.0 for cls in INTERESTING_CLASSES}, "found": False}
}
global_lcd_lock = threading.Lock() # Lock for safe LCD access
camera_threads = [] # List to keep track of camera threads
lcd_display_thread = None # Thread for LCD updates
terminate_threads_event = threading.Event() # Event to signal threads to stop

# GPIO & LCD setup
led_status = LED(LED_PIN)
led_status.off()
lcd = None
try:
    lcd = CharLCD('PCF8574', 0x27, cols=LCD_COLS, rows=LCD_ROWS)
    lcd.clear()
    lcd.write_string("Starting in ")  # Initial message
except Exception as e:
    print(f"LCD initialization failed: {e}")
    lcd = None  # Ensure safe fallback





# --- Signal Handler for Graceful Exit ---
def signal_handler(sig, frame):
    print("\nExiting gracefully...")
    # Signal all threads to terminate
    terminate_threads_event.set()

    # Join camera threads to ensure they finish gracefully
    for t in camera_threads:
        if t.is_alive():
            t.join(timeout=2) # Give thread a chance to finish

    # Join LCD display thread
    if lcd_display_thread and lcd_display_thread.is_alive():
        lcd_display_thread.join(timeout=2)

    if lcd:
        try:
            lcd.clear()
            lcd.write_string("Shutting Down...")
            time.sleep(1)
            lcd.clear()
        except Exception as lcd_err:
            print(f"LCD error during shutdown: {lcd_err}")

    cv2.destroyAllWindows()
    if 'led_status' in globals() and led_status is not None:
        led_status.off()

    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)





# --- Startup Countdown (Camera Warmup) ---
def startup_countdown(seconds=30):
    if lcd:
        try:
            lcd.clear()
            for i in range(seconds, 0, -1):
                lcd.home()
                lcd.write_string(f"Starting in {i}s")
                if i==10:
                    time.sleep(1)
                    lcd.clear()
                if i!=10:
                    time.sleep(1)
            lcd.clear()
            lcd.write_string("Initializing")
            time.sleep(1) # Give a moment for "Initializing" to be seen
            lcd.clear()
        except Exception as lcd_err:
            print(f"LCD error: {lcd_err}")
    else:
        print("LCD Error (LCD not initialized)") # Changed "LCD Error" for clarity
        for i in range(seconds, 0, -1):
            print(f"Starting in {i}s")
            time.sleep(1)









# --- Thread function for processing each camera stream ---
def process_camera_stream(video_path, cam_name, cam_id):
    cap_local = None
    try:
        cap_local = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap_local.isOpened():
            raise RuntimeError(f"Could not open RTSP stream for {cam_name}")
        cap_local.set(cv2.CAP_PROP_FPS, 24)
        cap_local.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception as e:
        print(f"[{cam_name}] Failed to open stream: {e}")
        return # Exit thread if stream cannot be opened

    frame_count = 0
    retry_count = 0
    max_retries = 5
    retry_delay = 1
    last_flag_state = False

    flag_path = f"detect_flag.txt"
    # Ensure flag file exists and is initialized
    with open(flag_path, "w") as f:
        f.write("0")

    while not terminate_threads_event.is_set(): # Check termination event

        ret, frame = cap_local.read()

        if not ret:
            print(f"[{cam_name}] Warning: Lost connection. Retrying...")
            retry_count += 1
            if retry_count <= max_retries:
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 10) # Exponential backoff, max 10s
                if cap_local is not None:
                    cap_local.release()
                cap_local = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
                if cap_local.isOpened():
                    cap_local.set(cv2.CAP_PROP_FPS, 24)
                    cap_local.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    print(f"[{cam_name}] Reconnected successfully.")
                    retry_count = 0 # Reset on successful reconnect
                    retry_delay = 1
                continue # Try reading frame again after reconnect
            else:
                print(f"[{cam_name}] Could not reconnect after multiple attempts. Terminating stream processing.")
                break # Exit loop, effectively ending the thread

        retry_count = 0 # Reset on successful frame read
        retry_delay = 1

        frame_count += 1
        if frame_count % FRAME_SKIP_INTERVAL != 0:
            time.sleep(0.01) # Small sleep to reduce CPU load
            continue
        results = model.predict(frame, imgsz=INFERENCE_IMG_SIZE, conf=MODEL_CONF_THRESHOLD, iou=MODEL_IOU_THRESHOLD, verbose=False) # Suppress verbose output
        detection_found_current_frame = False
        current_vehicle_count = {cls: 0 for cls in INTERESTING_CLASSES}
        current_vehicle_confidence = {cls: 0.0 for cls in INTERESTING_CLASSES}

        for res in results:
            boxes = res.boxes.xyxy.cpu().numpy().astype(int)
            classes = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            for i in range(len(boxes)):
                class_id = classes[i]
                conf = confs[i]
                if class_id in INTERESTING_CLASSES and conf > MODEL_CONF_THRESHOLD:
                    current_vehicle_count[class_id] += 1
                    detection_found_current_frame = True
                    if conf > current_vehicle_confidence[class_id]:
                        current_vehicle_confidence[class_id] = conf

        # Update global detection status safely
        with global_lcd_lock:
            global_detections[f"cam{cam_id}"]["count"] = current_vehicle_count
            global_detections[f"cam{cam_id}"]["confidence"] = current_vehicle_confidence
            global_detections[f"cam{cam_id}"]["found"] = detection_found_current_frame

        # Update individual camera's detection flag file
        current_flag_state = detection_found_current_frame
        if current_flag_state and not last_flag_state:
            with open(flag_path, "w") as f:
                f.write("1")
        elif not current_flag_state and last_flag_state:
            with open(flag_path, "w") as f:
                f.write("0")
        last_flag_state = current_flag_state

    if cap_local is not None:
        cap_local.release()
    print(f"[{cam_name}] Thread terminated.")








# --- Thread function for updating the LCD ---
def update_lcd_display():
    last_cam_displayed = 0 # 0 for none, 1 for cam1, 2 for cam2
    #toggle_interval = 2 # seconds to switch between camera displays if both are active

    while not terminate_threads_event.is_set():
        if lcd is None:
            print("LCD not initialized for display thread.")
            time.sleep(1)
            continue

        with global_lcd_lock:
            cam1_found = global_detections["cam1"]["found"]
            cam2_found = global_detections["cam2"]["found"]
            cam1_count = global_detections["cam1"]["count"]
            cam1_confidence = global_detections["cam1"]["confidence"]
            cam2_count = global_detections["cam2"]["count"]
            cam2_confidence = global_detections["cam2"]["confidence"]

        try:
            current_display_content = []
            update_lcd = 1
            if cam1_found and cam2_found:
                # Both cameras detect, alternate display
                if last_cam_displayed == 1:
                    target_cam_id = 2
                    target_cam_name = "Camera 2"
                    target_count = cam2_count
                    target_confidence = cam2_confidence
                    last_cam_displayed = 2
                else: # Default to cam1 or switch from cam2 to cam1
                    target_cam_id = 1
                    target_cam_name = "Camera 1"
                    target_count = cam1_count
                    target_confidence = cam1_confidence
                    last_cam_displayed = 1
                
                current_display_content.append(f"{target_cam_name}: Detected")
                for cid in INTERESTING_CLASSES:
                    if target_count[cid] > 0:
                        conf_percent = int(target_confidence[cid] * 100)
                        current_display_content.append(f"{class_names[cid]}:{target_count[cid]} {conf_percent}%")
                update_lcd = 1

            elif cam1_found:
                # Only Camera 1 detects
                last_cam_displayed = 1
                current_display_content.append("Camera 1: Detected")
                for cid in INTERESTING_CLASSES:
                    if cam1_count[cid] > 0:
                        conf_percent = int(cam1_confidence[cid] * 100)
                        current_display_content.append(f"{class_names[cid]}:{cam1_count[cid]} {conf_percent}%")
                update_lcd = 1

            elif cam2_found:
                # Only Camera 2 detects
                last_cam_displayed = 2
                current_display_content.append("Camera 2: Detected")
                for cid in INTERESTING_CLASSES:
                    if cam2_count[cid] > 0:
                        conf_percent = int(cam2_confidence[cid] * 100)
                        current_display_content.append(f"{class_names[cid]}:{cam2_count[cid]} {conf_percent}%")

            elif update_lcd == 1:
                # No detections
                last_cam_displayed = 0
                current_display_content.append("  Mauli Water Park  ")
                current_display_content.append("                    ")
                current_display_content.append("        Ready       ")
                current_display_content.append("Waiting for Vehicles")
                update_lcd = 0

            # Clear and write to LCD
            if update_lcd ==1:
                lcd.clear()
            
            for line_idx, text in enumerate(current_display_content):
                if line_idx < LCD_ROWS:
                    lcd.cursor_pos = (line_idx, 0)
                    lcd.write_string(text[:LCD_COLS]) # Truncate to fit LCD width

        except Exception as lcd_err:
            print(f"LCD display thread error: {lcd_err}")
        
        time.sleep(0.5)






# --- Main Application Logic ---
if __name__ == "__main__":
    startup_countdown(30) # Use the defined countdown function

    # --- Load YOLO Model ---
    try:
        model = YOLO("yash.onnx", task='detect')
        led_status.on()  # Success: LED ON
        if lcd:
            try:
                with global_lcd_lock: # Acquire lock before LCD operations
                    lcd.clear()
                    lcd.write_string("Model Loaded OK")
                time.sleep(1) # Display for a moment
                lcd.clear()
            except Exception as e:
                print(f"LCD initialization failed during model load: {e}")
        print("Model Loaded OK") # Also print to console
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")

    # RTSP URLs for two cameras
    cam1_url = 'rtsp://admin:mauli1234@192.168.1.64:554/Streaming/Channels/101'
    cam2_url = 'rtsp://admin:mauli1234@192.168.1.65:554/Streaming/Channels/101'

    # Create and start camera threads
    camera_threads.append(threading.Thread(target=process_camera_stream, args=(cam1_url, "Camera 1", 1), daemon=True))
    camera_threads.append(threading.Thread(target=process_camera_stream, args=(cam2_url, "Camera 2", 2), daemon=True))

    for t in camera_threads:
        t.start()
    print("Camera threads started.")

    # Start LCD update thread
    lcd_display_thread = threading.Thread(target=update_lcd_display, daemon=True)
    lcd_display_thread.start()
    print("LCD display thread started.")

    try:
        # Keep the main thread alive to allow daemon threads to run
        while True:
            time.sleep(2) # Main thread sleeps, allowing camera and LCD threads to work
            if terminate_threads_event.is_set():
                break # Exit main loop if termination is requested

    except Exception as e:
        print(f"CRITICAL SYSTEM FAILURE in main loop: {e}")
        terminate_threads_event.set() # Signal threads to stop on unexpected error
        # Re-raise the exception to go to the outer except block for system halt/reboot
        raise










try:
    yash =1
# --- Critical Error Handling Loop (Outer) ---
except Exception as e:
    print(f"CRITICAL SYSTEM FAILURE: {e}")
    # Ensure all threads are signaled to stop
    terminate_threads_event.set()

    # Attempt to join threads
    for t in camera_threads:
        if t.is_alive():
            t.join(timeout=2)
    if lcd_display_thread and lcd_display_thread.is_alive():
        lcd_display_thread.join(timeout=2)

    # This loop will run forever, displaying the error and preventing exit
    while True:
        if lcd:
            try:
                led_status.off() # Turn off LED
                with global_lcd_lock: # Acquire lock for LCD operations
                    lcd.clear()
                    lcd.cursor_pos = (0, 0)
                    lcd.write_string("CRITICAL ERROR")
                    lcd.cursor_pos = (1, 0)
                    lcd.write_string("SYSTEM HALTED")
                    lcd.cursor_pos = (2, 0)
                    error_type_name = type(e).__name__
                    if len(error_type_name) > LCD_COLS:
                        error_type_name = error_type_name[:LCD_COLS - 3] + "..."
                    lcd.write_string(f"[{error_type_name}]")
            except Exception as lcd_err:
                print(f"ERROR: Could not update LCD during critical error: {lcd_err}")
        print(f"CRITICAL ERROR: System Halted due to {type(e).__name__}: {e}")
        time.sleep(5) # Display for a few seconds

        if lcd:
            try:
                led_status.off() # Turn off LED
                # Display restart message and countdown
                for i in range(600, 0, -1):
                    with global_lcd_lock: # Acquire lock for LCD operations
                        lcd.home()
                        lcd.cursor_pos = (0, 0)
                        lcd.write_string("   SELF REPARING")
                        lcd.cursor_pos = (1, 0)
                        lcd.write_string(" SYSTEM RESTARTING")
                        lcd.cursor_pos = (2, 0)
                        lcd.write_string(f"       in {i}s")
                    time.sleep(1)
                with global_lcd_lock: # Acquire lock for LCD operations
                    lcd.clear()
                    lcd.write_string("Rebooting Now...")
                time.sleep(1)
                with global_lcd_lock: # Acquire lock for LCD operations
                    lcd.clear()
            except Exception as lcd_err:
                print(f"ERROR: Could not update LCD during critical error: {lcd_err}")

        # Attempt to reboot the system
        print("Initiating system reboot...")
        try:
            subprocess.run(["sudo", "reboot"], check=True)
        except Exception as reboot_err:
            print(f"Failed to initiate reboot: {reboot_err}")
            print("Please manually reboot the Raspberry Pi.")

        while True:
            if lcd:
                try:
                    with global_lcd_lock: # Acquire lock for LCD operations
                        lcd.home()
                        lcd.write_string("MANUAL REBOOT REQ!")
                        lcd.cursor_pos = (1,0)
                        lcd.write_string("Check console for")
                        lcd.cursor_pos = (2,0)
                        lcd.write_string("errors.")
                    time.sleep(1000000)
                except Exception as final_lcd_err:
                    print(f"Final LCD error: {final_lcd_err}")
            else:
                print("Manual reboot required! Check console for errors.")
                time.sleep(5) # Add a small sleep when no LCD