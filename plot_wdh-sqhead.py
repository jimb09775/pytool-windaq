"""
plot_wdh.py

Opens a WinDAQ .WDH file and plots:
  X axis — stringpot position in mm  : (ch2_volts - STRINGPOT_OFFSET_V) * 1000
  Y axis — sensor error in mm        : stringpot_mm - sensor_mm

Where:
  ch1 = channel 1 (radar/sensor,    4-20 mA → 1–5 V loop)
  ch2 = channel 2 (stringpot ref,   4-20 mA → 1–5 V loop)

Both channels are 4-20 mA current loops sharing the same nominal 1.0 V zero
at 4 mA, but physical offsets differ slightly due to shunt/trim variation:

  stringpot_mm = (ch2_volts - STRINGPOT_OFFSET_V) * 1000
  sensor_mm    = (ch1_volts - SENSOR_OFFSET_V)    * 1000
  error_mm     = stringpot_mm - sensor_mm

SENSOR_OFFSET_V is auto-detected from the file minimum (the sample where the
cylinder is fully retracted and the sensor reads 0 mm).  Override it with the
constant below if you prefer a fixed value.

WDH format notes (reverse-engineered from TESTNEW.WDH):
  - True data starts at byte offset 1160 (header is larger than the
    300-byte field in the file suggests due to extended channel descriptors)
  - 2-channel interleaved uint16, little-endian unsigned
  - ADC full scale: 10 V, unipolar positive (0–32768 counts = 0–10 V)
  - Voltage formula:  V = count / 32768 * 10
  - Last 5 words are a WinDAQ end-of-file marker and are discarded

Usage:
    python plot_wdh.py              # opens file picker
    python plot_wdh.py myfile.WDH   # direct path
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Configuration ──────────────────────────────────────────────────────────────
ADC_FULL_SCALE_V   = 10.0    # ADC input range (V)
ADC_COUNTS_FS      = 32768   # counts at full scale (unipolar positive half of int16)
STRINGPOT_OFFSET_V = 1.045   # ch2 voltage at stringpot 0 mm — squarehead
SENSOR_OFFSET_V    = 1.000   # ch1 voltage at sensor 0 mm (~252 Ω shunt)
                             # set to None to auto-detect from file's ch1 minimum
                             # (only reliable if the file starts fully retracted)
DATA_OFFSET_BYTES  = 1160    # byte offset where sample data begins
FOOTER_WORDS       = 5       # end-of-file marker words to discard
# ──────────────────────────────────────────────────────────────────────────────


def pick_file() -> str:
    """Open a file-picker dialog and return the selected path."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(
        title="Select WinDAQ .WDH file",
        filetypes=[("WinDAQ files", "*.WDH *.wdh"), ("All files", "*.*")]
    )
    root.destroy()
    return path


def load_wdh(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a WDH file and return (x_mm, y_mm).

    x_mm  — stringpot position (mm)
    y_mm  — stringpot minus sensor error (mm), positive = sensor reads short
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    # Parse interleaved uint16 samples, strip footer
    raw = np.frombuffer(data[DATA_OFFSET_BYTES:], dtype='<u2')
    raw = raw[:-FOOTER_WORDS]

    if len(raw) % 2 != 0:
        raw = raw[:-1]  # trim stray sample if present

    ch1 = raw[0::2].astype(np.float64)   # sensor / radar
    ch2 = raw[1::2].astype(np.float64)   # stringpot reference

    # Convert counts → volts
    scale = ADC_FULL_SCALE_V / ADC_COUNTS_FS
    v1 = ch1 * scale
    v2 = ch2 * scale

    # Sensor zero: use fixed constant or auto-detect from file minimum
    sensor_offset = SENSOR_OFFSET_V if SENSOR_OFFSET_V is not None else v1.min()

    # Convert volts → mm (each channel has its own 4-20mA zero offset)
    x_mm      = (v2 - STRINGPOT_OFFSET_V) * 1000.0   # stringpot position
    sensor_mm = (v1 - sensor_offset)      * 1000.0   # sensor position (0 = retracted)
    y_mm      = x_mm - sensor_mm                      # error: positive = sensor reads short

    return x_mm, y_mm


def plot(filepath: str) -> None:
    x_mm, y_mm = load_wdh(filepath)

    title = os.path.basename(filepath)
    n = len(x_mm)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle(f'WinDAQ Calibration Data — {title}', fontsize=13, fontweight='bold')

    # ── Top: scatter / line — error vs position ────────────────────────────────
    ax = axes[0]
    ax.scatter(x_mm, y_mm, s=1, alpha=0.4, color='steelblue', rasterized=True)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Stringpot Position (mm)')
    ax.set_ylabel('Stringpot − Sensor  (mm)')
    ax.set_title(f'Sensor error vs position  —  {n} samples')
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which='major', linestyle='-',  alpha=0.4)
    ax.grid(True, which='minor', linestyle=':', alpha=0.25)

    # Annotate stats
    valid = np.isfinite(y_mm)
    stats = (f'mean={y_mm[valid].mean():+.2f} mm   '
             f'std={y_mm[valid].std():.2f} mm   '
             f'min={y_mm[valid].min():+.2f}   max={y_mm[valid].max():+.2f}')
    ax.text(0.01, 0.97, stats, transform=ax.transAxes,
            fontsize=8, va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # ── Bottom: both channels vs sample index (time) ───────────────────────────
    t = np.arange(n)
    ax2 = axes[1]
    ax2.plot(t, x_mm, linewidth=0.8, color='darkorange', label='Ch2 — Stringpot (mm)')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Position (mm)')
    ax2.set_title('Stringpot position over time')
    ax2_r = ax2.twinx()
    ax2_r.plot(t, y_mm, linewidth=0.6, color='steelblue', alpha=0.7, label='Ch1−Ch2 error (mm)')
    ax2_r.set_ylabel('Error (mm)', color='steelblue')
    ax2_r.tick_params(axis='y', labelcolor='steelblue')
    ax2.grid(True, linestyle=':', alpha=0.4)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='best')

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = pick_file()

    if not filepath:
        print("No file selected.")
        return

    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    print(f"Loading: {filepath}")
    plot(filepath)


if __name__ == '__main__':
    main()
