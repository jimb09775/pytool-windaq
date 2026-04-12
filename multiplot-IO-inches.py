"""
multiplot-IO-inches.py

Opens one or more WinDAQ .WDH files and overlays all on a single
error-vs-position scatter plot, with extend and retract strokes
plotted separately so hysteresis is visible.

  X axis -- stringpot position in inches : (ch2_volts - STRINGPOT_OFFSET_V) * MM_TO_IN * 1000
  Y axis -- sensor error in inches       : stringpot_in - sensor_in

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
MM_TO_IN            = 1.0 / 25.4   # conversion factor
Y_SCALE_IN          = 1.0          # fixed Y axis range: +/- 1 inch
MEAN_BIN_MM         = 2.0     # x-axis bin width for mean trend line (mm, converted internally)
MEAN_MIN_SAMPLES    = 5       # minimum samples per bin to draw a mean point
MIN_POSITION_MM     = 35.0    # ignore samples below this position for max-error stats (mm, converted internally)

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

    # Convert to inches
    x_mm = x_mm * MM_TO_IN
    sensor_mm = sensor_mm * MM_TO_IN
    y_mm = y_mm * MM_TO_IN

    # Auto-scaled deadband: fraction of the 99th-percentile velocity magnitude.
    # Using percentile (not max) avoids ADC quantization spikes inflating the
    # deadband and excluding nearly all samples in slow or short-stroke files.
    x_smooth = smooth(x_mm, SMOOTH_WINDOW)
    velocity = np.gradient(x_smooth)
    deadband = DEADBAND_FRACTION * np.percentile(np.abs(velocity), 99)

    extend_mask  = velocity >  deadband
    retract_mask = velocity < -deadband

    return x_mm, y_mm, extend_mask, retract_mask


def mean_trend(x_in, y_in, bin_in=MEAN_BIN_MM * MM_TO_IN, min_samples=MEAN_MIN_SAMPLES):
    """
    Bin x_in into fixed-width buckets and return (bin_centers, bin_means)
    for bins that contain at least min_samples finite points.
    """
    ok = np.isfinite(x_in) & np.isfinite(y_in)
    x, y = x_in[ok], y_in[ok]
    if len(x) == 0:
        return np.array([]), np.array([])

    x_min = np.floor(x.min() / bin_in) * bin_in
    x_max = np.ceil(x.max()  / bin_in) * bin_in
    edges = np.arange(x_min, x_max + bin_in, bin_in)

    centers, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.sum() >= min_samples:
            centers.append((lo + hi) / 2.0)
            means.append(np.mean(y[mask]))

    return np.array(centers), np.array(means)


def make_plot(title, datasets, all_data=False):
    """
    Draw one scatter plot window.
    datasets: list of (label, x_mm, y_mm, color, legend_label)
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    summary_lines = []
    for label, x_mm, y_mm, color, leg in datasets:
        ok = np.isfinite(y_mm)
        if ok.sum() == 0:
            continue
        ax.scatter(x_mm, y_mm, s=1, alpha=0.4, color=color,
                   rasterized=True, label=leg)

        # Per-dataset mean trend line
        bx, by = mean_trend(x_mm, y_mm)
        if len(bx) > 1:
            ax.plot(bx, by, color=color, linewidth=1.2, alpha=0.85,
                    linestyle='-', zorder=3)

        mean = y_mm[ok].mean(); std = y_mm[ok].std()
        mn = y_mm[ok].min();    mx  = y_mm[ok].max()
        n  = ok.sum()
        summary_lines.append(
            f'{leg}:  mean={mean:+.4f}  std={std:.4f}  '
            f'min={mn:+.4f}  max={mx:+.4f}  n={n}  (in)'
        )

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Stringpot Position (in)')
    ax.set_ylabel('Stringpot - Sensor  (in)')
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which='major', linestyle='-',  alpha=0.4)
    ax.grid(True, which='minor', linestyle=':', alpha=0.25)
    ax.set_ylim(-Y_SCALE_IN, Y_SCALE_IN)

    stats_text = '\n'.join(summary_lines)
    ax.text(0.01, 0.99, stats_text, transform=ax.transAxes,
            fontsize=7.5, va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
    ax.legend(markerscale=6, fontsize=8, loc='lower right', framealpha=0.8)
    plt.tight_layout()
    plt.show()


def make_normalized_plot(title, datasets, show_legend=True):
    """
    4th plot: subtract each dataset's overall mean so all curves are centred on
    zero, then report max absolute deviation from zero for positions > MIN_POSITION_MM.
    datasets: list of (label, x_mm, y_mm, color, legend_label)
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    summary_lines = []
    for label, x_mm, y_mm, color, leg in datasets:
        ok = np.isfinite(y_mm)
        if ok.sum() == 0:
            continue

        # Subtract overall mean to normalise out calibration offset
        dataset_mean = y_mm[ok].mean()
        y_norm = y_mm - dataset_mean

        ax.scatter(x_mm, y_norm, s=1, alpha=0.4, color=color,
                   rasterized=True, label=leg)

        # Mean trend line on normalised data
        bx, by = mean_trend(x_mm, y_norm)
        if len(bx) > 1:
            ax.plot(bx, by, color=color, linewidth=1.2, alpha=0.85,
                    linestyle='-', zorder=3)

        # Max absolute deviation from zero, positions > MIN_POSITION_MM only
        pos_mask = (x_mm >= MIN_POSITION_MM * MM_TO_IN) & np.isfinite(y_norm)
        if pos_mask.sum() > 0:
            abs_dev   = np.abs(y_norm[pos_mask])
            max_dev   = abs_dev.max()
            worst_pos = x_mm[pos_mask][np.argmax(abs_dev)]
            worst_val = y_norm[pos_mask][np.argmax(abs_dev)]
            summary_lines.append(
                f'{leg}:  removed_mean={dataset_mean:+.4f}  '
                f'max_dev={max_dev:.4f} in  at {worst_pos:.3f} in ({worst_val:+.4f})'
            )

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Stringpot Position (in)')
    ax.set_ylabel('Error − mean  (in)')
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which='major', linestyle='-',  alpha=0.4)
    ax.grid(True, which='minor', linestyle=':', alpha=0.25)
    ax.set_ylim(-Y_SCALE_IN, Y_SCALE_IN)

    if show_legend:
        stats_text = '\n'.join(summary_lines)
        ax.text(0.01, 0.99, stats_text, transform=ax.transAxes,
                fontsize=7.5, va='top', family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
        ax.legend(markerscale=6, fontsize=8, loc='lower right', framealpha=0.8)
    plt.tight_layout()
    plt.show()


def multiplot(filepaths):
    base_colors = plt.get_cmap('tab10')

    all_ds  = []   # combined (retract + extend)
    ret_ds  = []   # retract only
    ext_ds  = []   # extend only

    for i, fp in enumerate(filepaths):
        label = os.path.basename(fp)
        c_ret = base_colors(i % 10)
        r, g, b, _ = c_ret
        c_ext = ((r + 1) / 2, (g + 1) / 2, (b + 1) / 2, 1.0)

        x_mm, y_mm, ext_mask, ret_mask = load_wdh(fp)

        for mask, tag in [(ret_mask, 'RET'), (ext_mask, 'EXT')]:
            vals = y_mm[mask]; ok = np.isfinite(vals)
            if ok.sum() == 0: continue
            mean = vals[ok].mean(); std = vals[ok].std()
            mn = vals[ok].min();    mx  = vals[ok].max(); n = ok.sum()
            print(f'{label} {tag}: mean={mean:+.4f} in  std={std:.4f} in  '
                  f'min={mn:+.4f}  max={mx:+.4f}  n={n}')

        all_ds.append((label, x_mm[ret_mask], y_mm[ret_mask], c_ret, f'{label} — retract'))
        all_ds.append((label, x_mm[ext_mask], y_mm[ext_mask], c_ext, f'{label} — extend'))
        ret_ds.append((label, x_mm[ret_mask], y_mm[ret_mask], c_ret, label))
        ext_ds.append((label, x_mm[ext_mask], y_mm[ext_mask], c_ext, label))

    make_plot('All Data — Extend + Retract', all_ds)
    make_plot('Retract Only', ret_ds)
    make_plot('Extend Only',  ext_ds)
    make_normalized_plot('Mean-Normalised — All Data (max dev >35 mm shown, inches)', all_ds)
    make_normalized_plot('Mean-Normalised — All Data (no legend)', all_ds, show_legend=False)


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
