# gpio_patch.py
from gpiozero import LED
import sys

class GPIOShim:
    BCM = 'BCM'
    OUT = 'OUT'
    LOW = 0
    HIGH = 1

    def __init__(self):
        self._pins = {}

    def setmode(self, mode):
        # Accept but ignore for compatibility
        pass

    def setup(self, pin, mode):
        if mode == self.OUT:
            self._pins[pin] = LED(pin)

    def output(self, pin, value):
        led = self._pins.get(pin)
        if led:
            if value == self.HIGH:
                led.on()
            else:
                led.off()

    def cleanup(self):
        for led in self._pins.values():
            led.off()
        self._pins.clear()

GPIO = GPIOShim()
sys.modules['RPi.GPIO'] = GPIO
