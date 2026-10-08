"""Optional direct-GPIO LCD1602 status output for the Pi detector."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any


LCD_COLUMNS = 16
LCD_ROWS = 2
DISPLAY_LINES = {
    "nominal": ("NORMAL OPERATION", ""),
    "A": ("IMPLEMENT", "ATTACHED"),
    "B": ("FLAT TIRE", ""),
    "AB": ("IMPLEMENT", "FLAT TIRE"),
}


def display_lines(class_name: str) -> tuple[str, str]:
    """Return two LCD-safe lines for a detector class."""
    try:
        return DISPLAY_LINES[class_name]
    except KeyError as exc:
        raise ValueError(f"unsupported detector class {class_name!r}") from exc


def _fit_line(line: str) -> str:
    """Pad or trim a line to the LCD1602's 16-character width."""
    return line[:LCD_COLUMNS].ljust(LCD_COLUMNS)


class NullLCD:
    """No-op display used when LCD output is not enabled."""

    def show_class(self, class_name: str) -> None:
        del class_name

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class LCDPinout:
    """BCM GPIO mapping for a direct 8-bit LCD1602 connection."""

    rs: int = 15
    rw: int = 18
    enable: int = 23
    d0: int = 2
    d1: int = 3
    d2: int = 4
    d3: int = 17
    d4: int = 27
    d5: int = 22
    d6: int = 10
    d7: int = 9

    @property
    def data(self) -> tuple[int, ...]:
        return self.d0, self.d1, self.d2, self.d3, self.d4, self.d5, self.d6, self.d7


class LCD1602:
    """Write detector status to a directly connected 8-bit HD44780 LCD1602."""

    def __init__(
        self,
        *,
        pinout: LCDPinout = LCDPinout(),
        gpio: Any | None = None,
    ) -> None:
        if gpio is None:
            try:
                import RPi.GPIO as gpio
            except ImportError as exc:
                raise RuntimeError(
                    "Direct LCD support requires RPi.GPIO; "
                    "install requirements-pi.txt on the Raspberry Pi"
                ) from exc
        self._gpio = gpio
        self._pinout = pinout
        self._setup_gpio()
        self._initialize_lcd()

    def _setup_gpio(self) -> None:
        gpio = self._gpio
        gpio.setmode(gpio.BCM)
        gpio.setup(
            [self._pinout.rs, self._pinout.rw, self._pinout.enable, *self._pinout.data],
            gpio.OUT,
            initial=gpio.LOW,
        )

    def _pulse_enable(self) -> None:
        self._gpio.output(self._pinout.enable, self._gpio.HIGH)
        time.sleep(0.000001)
        self._gpio.output(self._pinout.enable, self._gpio.LOW)
        time.sleep(0.0001)

    def _send_byte(self, value: int, *, character: bool) -> None:
        self._gpio.output(
            self._pinout.rs,
            self._gpio.HIGH if character else self._gpio.LOW,
        )
        self._gpio.output(self._pinout.rw, self._gpio.LOW)
        for pin, bit in zip(self._pinout.data, range(8)):
            self._gpio.output(
                pin,
                self._gpio.HIGH if value & (1 << bit) else self._gpio.LOW,
            )
        self._pulse_enable()
        if not character:
            time.sleep(0.00005)

    def _initialize_lcd(self) -> None:
        time.sleep(0.05)
        for value in (0x30, 0x30, 0x30):
            self._send_byte(value, character=False)
            time.sleep(0.005)
        self._send_byte(0x38, character=False)
        self._send_byte(0x0C, character=False)
        self._send_byte(0x06, character=False)
        self._send_byte(0x01, character=False)
        time.sleep(0.002)

    def show_class(self, class_name: str) -> None:
        lines = display_lines(class_name)
        self._send_byte(0x01, character=False)
        time.sleep(0.002)
        for row, line in enumerate(lines):
            self._send_byte(0x80 | (0x40 * row), character=False)
            for character in _fit_line(line):
                self._send_byte(ord(character), character=True)

    def close(self) -> None:
        self._gpio.cleanup([
            self._pinout.rs,
            self._pinout.rw,
            self._pinout.enable,
            *self._pinout.data,
        ])


def create_lcd(*, enabled: bool, pinout: LCDPinout = LCDPinout()):
    """Create the configured display, or a no-op when LCD output is disabled."""
    if not enabled:
        return NullLCD()
    return LCD1602(pinout=pinout)
