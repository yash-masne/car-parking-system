import time
import os # Import os for file existence check
from gpiozero import LED # Import LED from gpiozero for pin control

# GPIO setup
LED_PIN = 22
SIGNAL_PIN = 26  # Signal (Boom Barrier Open)
HEARTBEAT_LED = 27  # Stays ON if code is healthy

# Initialize the LED objects using gpiozero
# gpiozero handles setting the mode and initial state when the object is created
signal_pin_control = LED(SIGNAL_PIN)
signal_pin_control.off() # Equivalent to GPIO.output(SIGNAL_PIN, GPIO.LOW)

led_pin_control = LED(LED_PIN)
led_pin_control.off() # Equivalent to GPIO.output(LED_PIN, GPIO.LOW)

heartbeat_led_control = LED(HEARTBEAT_LED)
heartbeat_led_control.on() # Equivalent to GPIO.output(HEARTBEAT_LED, GPIO.HIGH)


# Initialize prev_flag based on current file content if it exists, otherwise "0"
prev_flag = "0"

# Function to control the boom barrier
def control_signal():
    flag_file_path = "detect_flag.txt"
    if os.path.exists(flag_file_path):
        try:
            with open(flag_file_path, "r") as f:
                prev_flag = f.read().strip()
        except Exception as e:
            print(f"Error reading initial flag file: {e}")
            prev_flag = "0"

    print(f"Initial prev_flag: {prev_flag}")

    while True:
        if prev_flag ==0:
            current_flag = 0
        try:
            with open(flag_file_path, "r") as f:
                content = f.read().strip()
                if content: # Ensure content is not empty
                    current_flag = content
                heartbeat_led_control.on() # Keep heartbeat on if file read successfully
        except FileNotFoundError:
            print("Flag file not found. Ensure onnx.py is running and creating it.")
            heartbeat_led_control.off()  # Turn OFF on persistent file error
            time.sleep(5) # Wait before retrying to prevent busy-looping
            continue # Try again
        except Exception as e:
            print(f"Error reading flag file: {e}")
            heartbeat_led_control.off()
            time.sleep(5)
            continue

        # --- Logic for triggering boom barrier ---
        # Condition: Vehicle detected (current_flag is "1") AND it was not detected before (prev_flag was "0")
        if current_flag == "1" and prev_flag == "0":

            print("Vehicle Present: Triggering boom barrier.")
            signal_pin_control.on()  # Activate Signal (Boom Barrier Open)
            print("Signal High")

            # Blinking LED while gate is opening (e.g., 3 seconds)
            blink_end_time = time.time() + 3
            while time.time() < blink_end_time:
                led_pin_control.on()
                time.sleep(0.5)
                led_pin_control.off()
                time.sleep(0.5)
            led_pin_control.off() # Ensure LED is off after blinking

            # --- Logic for waiting for vehicle to leave and closing barrier ---
            print("Boom barrier opened. Waiting for vehicle to clear...")
            # Keep the barrier open as long as '1' is detected.
            # The barrier should only close *after* the vehicle leaves.
            vehicle_present = True
            while vehicle_present:
                try:
                    with open(flag_file_path, "r") as f:
                        current_flag_in_wait = f.read().strip()
                except FileNotFoundError:
                    # If file disappears while waiting, assume vehicle has left or error occurred
                    current_flag_in_wait = "0"
                
                if current_flag_in_wait == "0":
                    print("No vehicle detected. Closing boom barrier.")
                    signal_pin_control.off() # Deactivate Signal (Boom Barrier Close)
                    vehicle_present = False # Exit this inner loop
                else:
                    print("Vehicle still present. Boom barrier remains open.")
                    time.sleep(1) # Check again after a delay

            # Barrier is now closed. Add a cooldown period if desired.
            print("Boom barrier closed. Entering cooldown period.")
            blink_end_time = time.time() + 1
            while time.time() < blink_end_time:
                led_pin_control.on()
                time.sleep(0.5)
                led_pin_control.off()
                time.sleep(0.5)
            led_pin_control.off()
        # Always update prev_flag at the end of the main loop iteration
        # to reflect the state *before* the next iteration begins.
        
# Small delay to prevent busy-looping and reduce CPU usage
        time.sleep(0.2)

# Start loop
try:
    control_signal()
except KeyboardInterrupt:
    print("Stopped by user.")
    heartbeat_led_control.off() # Turn OFF on manual stop
    signal_pin_control.off()  # Turn OFF on manual stop
    led_pin_control.off()
finally:
    # gpiozero handles cleanup automatically on script exit.
    # No explicit cleanup() call needed.
    pass