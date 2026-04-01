"""
multiplot_wdh.py

Opens one or more WinDAQ .WDH files and overlays all on a single
error-vs-position scatter plot.

  X axis — stringpot position in mm  : (ch2_volts - STRINGPOT_OFFSET_V) * 1000
  Y axis — sensor error in mm        : stringpot_mm - sensor_mm

Where:
  ch1 = channel 1 (radar/sensor,    4-20 mA → 1–5 V loop)
  ch2 = channel 2 (stringpot ref,   4-20 mA → 1–5 V loop)

  stringpot_mm = (ch2_volts - STRINGPOT_OFFSET_V) * 1000
  sensor_mm    = (ch1_volts - SENSOR_OFFSET_V)    * 1000
  error_mm     = stringpot_mm - sensor_mm

SENSOR_OFFSET_V is auto-detected per file from its channel-1 minimum
(the sample where the cylinder is fully retracted and the sensor reads 0 mm).
Override with a fixed float below if preferred.

WDH format notes:
  - Data starts at byte offset 1160
  - 2-channel interleaved uint16, little-endian unsigned
  - ADC full scale: 10 V, unipolar (0–32768 counts = 0–10 V)
  - Last 5 words are a WinDAQ EOF marker and are discarded

Usage:
    python multiplot_wdh.py                        # opens multi-file picker
    python multiplot_wdh.py file1.WDH file2.WDH    # direct paths
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Configuration ──────────────────────────────────────────────────────────────
ADC_FULL_SCALE_V   = 10.0    # ADC input range (V)
ADC_COUNTS_FS      = 32768   # counts at full scale
STRINGPOT_OFFSET_V = 1.232   # ch2 voltage at stringpot 0 mm (measured, 250 Ω shunt)
SENSOR_OFFSET_V    = 1.000   # ch1 voltage at sensor 0 mm (different shunt on ch1)
                             # set to None to auto-detect from each file's ch1 minimum
                             # (only reliable if the file starts fully retracted)
DATA_OFFSET_BYTES  = 1160    # byte offset where sample data begins
FOOTER_WORDS       = 5       # EOF marker words to discard
Y_SCALE_MM         = 20.0    # fixed Y axis range: +/- this value in mm
# ──────────────────────────────────────────────────────────────────────────────


def pick_files() -> list[str]:
    """Open a multi-file picker dialog and return selected paths."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    paths = filedialog.askopenfilenames(
        title="Select WinDAQ .WDH files",
        filetypes=[("WinDAQ files", "*.WDH *.wdh"), ("All files", "*.*")]
    )
    root.destroy()
    return list(paths)


def load_wdh(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a WDH file and return (x_mm, y_mm).

    x_mm — stringpot position (mm)
    y_mm — error: stringpot − sensor (mm), positive = sensor reads short
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    raw = np.frombuffer(data[DATA_OFFSET_BYTES:], dtype='<u2')
    raw = raw[:-FOOTER_WORDS]
    if len(raw) % 2 != 0:
        raw = raw[:-1]

    ch1 = raw[0::2].astype(np.float64)   # sensor / radar
    ch2 = raw[1::2].astype(np.float64)   # stringpot reference

    scale = ADC_FULL_SCALE_V / ADC_COUNTS_FS
    v1 = ch1 * scale
    v2 = ch2 * scale

    sensor_offset = SENSOR_OFFSET_V if SENSOR_OFFSET_V is not None else v1.min()

    x_mm      = (v2 - STRINGPOT_OFFSET_V) * 1000.0
    sensor_mm = (v1 - sensor_offset)      * 1000.0
    y_mm      = x_mm - sensor_mm

    return x_mm, y_mm


def multiplot(filepaths: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.suptitle('WinDAQ Calibration — Sensor Error vs Position', fontsize=13, fontweight='bold')

    cmap = plt.get_cmap('tab10')
    summary_lines = []

    for i, fp in enumerate(filepaths):
        label = os.path.basename(fp)
        color = cmap(i % 10)

        x_mm, y_mm = load_wdh(fp)
        valid = np.isfinite(y_mm)

        ax.scatter(x_mm, y_mm, s=1, alpha=0.35, color=color,
                   rasterized=True, label=label)

        mean = y_mm[valid].mean()
        std  = y_mm[valid].std()
        mn   = y_mm[valid].min()
        mx   = y_mm[valid].max()
        n    = valid.sum()
        summary_lines.append(
            f'{label}:  mean={mean:+.2f}  std={std:.2f}  '
            f'min={mn:+.2f}  max={mx:+.2f}  n={n}'
        )
        print(f'{label}: mean={mean:+.2f} mm  std={std:.2f} mm  '
              f'min={mn:+.2f}  max={mx:+.2f}  n={n}')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Stringpot Position (mm)')
    ax.set_ylabel('Stringpot − Sensor  (mm)')
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which='major', linestyle='-',  alpha=0.4)
    ax.grid(True, which='minor', linestyle=':', alpha=0.25)
    ax.set_ylim(-Y_SCALE_MM, Y_SCALE_MM)

    stats_text = '\n'.join(summary_lines)
    ax.text(0.01, 0.99, stats_text, transform=ax.transAxes,
            fontsize=7.5, va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

    ax.legend(markerscale=6, fontsize=8, loc='lower right', framealpha=0.8)

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) > 1:
        filepaths = sys.argv[1:]
    else:
        filepaths = pick_files()

    if not filepaths:
        print("No files selected.")
        return

    missing = [f for f in filepaths if not os.path.isfile(f)]
    if missing:
        for f in missing:
            print(f"File not found: {f}")
        sys.exit(1)

    print(f"Loading {len(filepaths)} file(s)...")
    multiplot(filepaths)


if __name__ == '__main__':
    main()
