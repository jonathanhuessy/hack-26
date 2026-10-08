import unittest
from unittest.mock import patch

try:
    from .lcd_status import LCD1602, LCDPinout, NullLCD, display_lines
except ImportError:
    from lcd_status import LCD1602, LCDPinout, NullLCD, display_lines


class FakeGPIO:
    BCM = "BCM"
    OUT = "OUT"
    LOW = 0
    HIGH = 1

    def __init__(self):
        self.mode = None
        self.setup_args = None
        self.outputs = []
        self.cleaned = None

    def setmode(self, mode):
        self.mode = mode

    def setup(self, pins, mode, initial):
        self.setup_args = (pins, mode, initial)

    def output(self, pin, value):
        self.outputs.append((pin, value))

    def cleanup(self, pins):
        self.cleaned = pins


class LCDStatusTests(unittest.TestCase):
    def test_display_lines_map_detector_classes(self):
        self.assertEqual(display_lines("nominal"), ("NORMAL OPERATION", ""))
        self.assertEqual(display_lines("A"), ("IMPLEMENT", "ATTACHED"))
        self.assertEqual(display_lines("B"), ("FLAT TIRE", ""))
        self.assertEqual(display_lines("AB"), ("IMPLEMENT", "FLAT TIRE"))

    @patch("pi.lcd_status.time.sleep")
    def test_direct_gpio_display_uses_default_pinout(self, _sleep):
        gpio = FakeGPIO()
        display = LCD1602(gpio=gpio)

        self.assertEqual(gpio.mode, gpio.BCM)
        self.assertEqual(
            gpio.setup_args,
            ([15, 18, 23, 2, 3, 4, 17, 27, 22, 10, 9], gpio.OUT, gpio.LOW),
        )
        display.show_class("A")
        self.assertGreater(len(gpio.outputs), 0)

        display.close()
        self.assertEqual(gpio.cleaned, [15, 18, 23, 2, 3, 4, 17, 27, 22, 10, 9])

    @patch("pi.lcd_status.time.sleep")
    def test_direct_gpio_display_accepts_custom_pinout(self, _sleep):
        gpio = FakeGPIO()
        pinout = LCDPinout(
            rs=1,
            rw=2,
            enable=3,
            d0=4,
            d1=5,
            d2=6,
            d3=7,
            d4=8,
            d5=9,
            d6=10,
            d7=11,
        )

        LCD1602(gpio=gpio, pinout=pinout)

        self.assertEqual(gpio.setup_args[0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    def test_null_lcd_is_safe_without_hardware(self):
        display = NullLCD()
        display.show_class("nominal")
        display.close()

    def test_unknown_class_is_rejected(self):
        with self.assertRaises(ValueError):
            display_lines("unknown")


if __name__ == "__main__":
    unittest.main()
