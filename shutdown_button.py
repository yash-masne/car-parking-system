from gpiozero import Button
from signal import pause
import os
import subprocess
import signal
import time

# Try to import and initialize the LCD
try:
    from RPLCD.i2c import CharLCD
    lcd = CharLCD('PCF8574', 0x27)
    lcd_available = True
except Exception as e:
    print("LCD not found or failed to initialize:", e)
    lcd_available = False

# Shutdown handler
def shutdown():
    print("Shutdown button pressed! Powering off...")
    if lcd_available:
        try:
            lcd.clear()
            lcd.write_string("   Shutting down       Please wait...")
        except Exception as e:
            print("Failed to write to LCD:", e)
    
    
    time.sleep(1)

    if lcd_available:
        try:
            lcd.clear()
            lcd.write_string("   TURN OFF MAIN           SWITCH")
        except Exception as e:
            print("Failed to write final LCD message:", e)

    # IMPORTANT: delay before shutdown so LCD can update
    time.sleep(1)
    os.system("sudo shutdown -h now")
    time.sleep(2)

# Button setup (GPIO 21)
shutdown_btn = Button(21, pull_up=False)
shutdown_btn.when_pressed = shutdown

print("Shutdown button monitoring started...")
pause()
