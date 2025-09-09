import os
import cv2
import threading
import time

# Set OpenCV FFMPEG options for RTSP
# Use TCP transport and set max_delay (in microseconds) to 1,000,000 (1 second)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp;max_delay;1000000"

class RTSPCamera:
    def __init__(self, rtsp_url, window_name):
        self.rtsp_url = rtsp_url
        self.window_name = window_name
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.running = True
        self.frame = None
        self.lock = threading.Lock()

        # Start background thread to read frames
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                print(f"⚠️ {self.window_name}: Frame read failed, retrying...")
                time.sleep(0.1)  # small delay before retrying

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()
        cv2.destroyWindow(self.window_name)


def main():
    cam1_url = 'rtsp://admin:mauli1234@192.168.1.64:554/Streaming/Channels/101'
    cam2_url = 'rtsp://admin:mauli1234@192.168.1.65:554/Streaming/Channels/101'

    cam1 = RTSPCamera(cam1_url, "Camera 1")
    cam2 = RTSPCamera(cam2_url, "Camera 2")

    print("📷 Streaming both cameras. Press 'q' to quit.")

    while True:
        frame1 = cam1.get_frame()
        frame2 = cam2.get_frame()

        if frame1 is not None:
            cv2.imshow("Camera 1", frame1)
        if frame2 is not None:
            cv2.imshow("Camera 2", frame2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting...")
            break

    cam1.stop()
    cam2.stop()

if __name__ == "__main__":
    main()
