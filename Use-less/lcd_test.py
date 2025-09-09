from RPLCD.i2c import CharLCD
import time

lcd = CharLCD('PCF8574', 0x27, cols=20, rows=4) # Use your detected address
lcd.clear()
lcd.write_string('Hello, Pi!')
time.sleep(2)
lcd.clear()
lcd.write_string('Testing 123')
time.sleep(2)
lcd.clear()