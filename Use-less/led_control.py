import gpiod
import time

# Define GPIO pin (BCM 17)
CHIP = "gpiochip0"  # Use the correct chip name from gpiodetect
LINE = 17           # BCM GPIO 17

# Setup GPIO line
chip = gpiod.Chip(CHIP)
line = chip.get_line(LINE)
line.request(consumer="led_control", type=gpiod.LINE_REQ_DIR_OUT)

# Function to control the LED based on the flag
def control_led():
    flag_file_path = "detect_flag.txt"
    
    while True:
        # Read the detection flag
        try:
            with open(flag_file_path, "r") as f:
                flag_value = f.read().strip()

            # Check flag value and control LED
            if flag_value == "1":
                line.set_value(1)  # Turn LED on
                print("Vehicle detected: LED ON")
            elif flag_value == "0":
                line.set_value(0)  # Turn LED off
                print("No vehicle detected: LED OFF")
            else:
                print("Invalid flag value, retrying...")
                time.sleep(1)  # Wait before retrying

        except FileNotFoundError:
            print("Flag file not found, retrying in 2 seconds...")
            time.sleep(2)  # Wait before retrying the file read
        except Exception as e:
            print(f"Unexpected error: {e}, retrying...")
            time.sleep(2)  # Wait before retrying

        time.sleep(1)  # Check every second

# Start controlling the LED based on the flag
try:
    control_led()
except KeyboardInterrupt:
    print("LED control stopped by user.")
finally:
    line.set_value(0)  # Turn off the LED before cleanup
    line.release()     # Release GPIO line
