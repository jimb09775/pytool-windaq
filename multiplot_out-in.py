"""
multiplot_IO.py

Opens one or more WinDAQ .WDH files and overlays all on a single
error-vs-position scatter plot, with extend and retract strokes
plotted separately so hysteresis is visible.

  X axis -- stringpot position in mm  : (ch2_volts - STRINGPOT_OFFSET_V) * 1000
  Y axis -- sensor error in mm        : stringpot_mm - sensor_mm

Where:
  ch1 = channel 1 (radar/sensor,    4-20 mA -> 1-5 V loop)
  ch2 = channel 2 (stringpot ref,   4-20 mA -> 1-5 V loop)

  stringpot_mm = (ch2_volts - STRINGPOT_OFFSET_V) * 1000
  sensor_mm    = (ch1_volts - SENSOR_OFFSET_V)    * 1000
  error_mm     = stringpot_mm - sensor_mm

Direction detection:
  The stringpot position is smoothed then differentiated.  The deadband
  threshold is set automatically to 10% of the peak velocity in each file,
  so it works regardless of sample rate or stroke speed.  Samples in the
  turnaround zone (|vel| < deadband) are excluded from both directions.

WDH format notes:
  - Data starts at byte offset 1160
  - 2-channel interleaved uint16, little-endian unsigned
  - ADC full scale: 10 V, unipolar (0-32768 counts = 0-10 V)
  - Last 5 words are a WinDAQ EOF marker and are discarded

Usage:
    python multiplot_IO.py                        # opens multi-file picker
    python multiplot_IO.py file1.WDH file2.WDH    # direct paths
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# -- Configuration -------------------------------------------------------------
ADC_FULL_SCALE_V    = 10.0    # ADC input range (V)
ADC_COUNTS_FS       = 32768   # counts at full scale
STRINGPOT_OFFSET_V  = 1.232   # ch2 voltage at stringpot 0 mm (250 ohm shunt)
SENSOR_OFFSET_V     = 1.000   # ch1 voltage at sensor 0 mm
                              # set to None to auto-detect from each file's ch1 min
DATA_OFFSET_BYTES   = 1160    # byte offset where sample data begins
FOOTER_WORDS        = 5       # EOF marker words to discard
Y_SCALE_MM          = 20.0    # fixed Y axis range: +/- this value in mm

# Direction detection
SMOOTH_WINDOW       = 51      # samples to smooth position before differentiating
DEADBAND_FRACTION   = 0.10    # fraction of peak velocity used as turnaround deadband
                              # 0.10 = exclude bottom 10% of velocity magnitude
# ------------------------------------------------------------------------------


def pick_files():
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


def smooth(x, window):
    w = min(window, len(x))
    if w % 2 == 0:
        w -= 1
    return np.convolve(x, np.ones(w) / w, mode='same')


def load_wdh(filepath):
    """
    Returns (x_mm, y_mm, extend_mask, retract_mask).
    extend_mask / retract_mask are bool arrays classifying each sample.
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

    # Auto-scaled deadband: fraction of peak velocity in this file
    x_smooth = smooth(x_mm, SMOOTH_WINDOW)
    velocity = np.gradient(x_smooth)
    deadband = DEADBAND_FRACTION * np.abs(velocity).max()

    extend_mask  = velocity >  deadband
    retract_mask = velocity < -deadband

    return x_mm, y_mm, extend_mask, retract_mask


def multiplot(filepaths):
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle('WinDAQ Calibration -- Sensor Error vs Position (Extend / Retract)',
                 fontsize=13, fontweight='bold')

    base_colors = plt.get_cmap('tab10')
    summary_lines = []

    for i, fp in enumerate(filepaths):
        label = os.path.basename(fp)
        c_ext = base_colors(i % 10)
        r, g, b, _ = c_ext
        c_ret = ((r + 1) / 2, (g + 1) / 2, (b + 1) / 2, 1.0)  # lightened

        x_mm, y_mm, ext_mask, ret_mask = load_wdh(fp)

        ax.scatter(x_mm[ext_mask], y_mm[ext_mask], s=1, alpha=0.4,
                   color=c_ext, rasterized=True, label=f'{label} — extend')
        ax.scatter(x_mm[ret_mask], y_mm[ret_mask], s=1, alpha=0.4,
                   color=c_ret, rasterized=True, label=f'{label} — retract')

        for mask, tag in [(ext_mask, 'EXT'), (ret_mask, 'RET')]:
            vals = y_mm[mask]
            ok = np.isfinite(vals)
            if ok.sum() == 0:
                continue
            mean = vals[ok].mean()
            std  = vals[ok].std()
            mn   = vals[ok].min()
            mx   = vals[ok].max()
            n    = ok.sum()
            summary_lines.append(
                f'{label} {tag}:  mean={mean:+.2f}  std={std:.2f}  '
                f'min={mn:+.2f}  max={mx:+.2f}  n={n}'
            )
            print(f'{label} {tag}: mean={mean:+.2f} mm  std={std:.2f} mm  '
                  f'min={mn:+.2f}  max={mx:+.2f}  n={n}')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Stringpot Position (mm)')
    ax.set_ylabel('Stringpot - Sensor  (mm)')
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
