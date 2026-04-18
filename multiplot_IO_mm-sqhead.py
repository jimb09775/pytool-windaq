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

Velocity-weighted average:
  A faster stroke produces a larger velocity-induced lag in the sensor
  output, making its mean error a less accurate estimate of the true
  static offset.  For each file the stats box and console output include
  a velocity-weighted average mean:

      weight = 1 / mean_speed_of_direction
      vel_wtd_avg = (mean_ext * w_ext + mean_ret * w_ret) / (w_ext + w_ret)

  This gives the slower (more accurate) direction proportionally more
  influence, yielding a tighter cross-unit spread than a simple average.
  Speeds are reported in ADC counts per sample; the ratio is what matters,
  not the absolute value.

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
STRINGPOT_OFFSET_V  = 1.045   # ch2 voltage at stringpot 0 mm — squarehead
SENSOR_OFFSET_V     = 1.000   # ch1 voltage at sensor 0 mm
                              # set to None to auto-detect from each file's ch1 min
DATA_OFFSET_BYTES   = 1160    # byte offset where sample data begins
FOOTER_WORDS        = 5       # EOF marker words to discard
Y_SCALE_MM          = 20.0    # fixed Y axis range: +/- this value in mm
MEAN_BIN_MM         = 2.0     # x-axis bin width for mean trend line (mm)
MEAN_MIN_SAMPLES    = 5       # minimum samples per bin to draw a mean point
MIN_POSITION_MM     = 35.0    # ignore samples below this position for max-error stats

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

    # Auto-scaled deadband: fraction of the 99th-percentile velocity magnitude.
    # Using percentile (not max) avoids ADC quantization spikes inflating the
    # deadband and excluding nearly all samples in slow or short-stroke files.
    x_smooth = smooth(x_mm, SMOOTH_WINDOW)
    velocity = np.gradient(x_smooth)
    deadband = DEADBAND_FRACTION * np.percentile(np.abs(velocity), 99)

    extend_mask  = velocity >  deadband
    retract_mask = velocity < -deadband

    return x_mm, y_mm, extend_mask, retract_mask, velocity


def mean_trend(x_mm, y_mm, bin_mm=MEAN_BIN_MM, min_samples=MEAN_MIN_SAMPLES):
    """
    Bin x_mm into fixed-width buckets and return (bin_centers, bin_means)
    for bins that contain at least min_samples finite points.
    """
    ok = np.isfinite(x_mm) & np.isfinite(y_mm)
    x, y = x_mm[ok], y_mm[ok]
    if len(x) == 0:
        return np.array([]), np.array([])

    x_min = np.floor(x.min() / bin_mm) * bin_mm
    x_max = np.ceil(x.max()  / bin_mm) * bin_mm
    edges = np.arange(x_min, x_max + bin_mm, bin_mm)

    centers, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.sum() >= min_samples:
            centers.append((lo + hi) / 2.0)
            means.append(np.mean(y[mask]))

    return np.array(centers), np.array(means)


def _vel_weighted_mean(y_mm_a, vel_a, y_mm_b, vel_b):
    """
    Return the velocity-weighted average of two directional means.

    A faster stroke introduces more velocity-induced lag, so it is a less
    accurate estimate of the true static position.  Weighting each direction's
    mean inversely by its average speed compensates for this: the slower
    (more accurate) direction contributes more to the combined estimate.

        w = 1 / mean_speed
        vel_wtd_avg = (mean_a * w_a + mean_b * w_b) / (w_a + w_b)
    """
    ok_a = np.isfinite(y_mm_a); ok_b = np.isfinite(y_mm_b)
    if ok_a.sum() == 0 or ok_b.sum() == 0:
        return None
    spd_a = np.abs(vel_a[ok_a]).mean()
    spd_b = np.abs(vel_b[ok_b]).mean()
    if spd_a == 0 or spd_b == 0:
        return None
    w_a = 1.0 / spd_a
    w_b = 1.0 / spd_b
    mean_a = y_mm_a[ok_a].mean()
    mean_b = y_mm_b[ok_b].mean()
    return (mean_a * w_a + mean_b * w_b) / (w_a + w_b)


def make_plot(title, datasets, all_data=False):
    """
    Draw one scatter plot window.
    datasets: list of (label, x_mm, y_mm, color, legend_label, velocity)

    When a file contributes both an extend and a retract dataset the stats box
    also shows the velocity-weighted average mean, which corrects for the fact
    that the faster retract stroke has a proportionally larger lag offset than
    the slower extend stroke.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Group entries by base label so we can pair ext/ret for the same file
    from collections import defaultdict
    by_label = defaultdict(list)

    summary_lines = []
    for label, x_mm, y_mm, color, leg, velocity in datasets:
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
            f'{leg}:  mean={mean:+.2f}  std={std:.2f}  '
            f'min={mn:+.2f}  max={mx:+.2f}  n={n}'
        )
        by_label[label].append((y_mm, velocity))

    # Append velocity-weighted average line for each file that has both directions
    for label, entries in by_label.items():
        if len(entries) == 2:
            (y_a, vel_a), (y_b, vel_b) = entries
            vwa = _vel_weighted_mean(y_a, vel_a, y_b, vel_b)
            if vwa is not None:
                summary_lines.append(
                    f'{label}  vel-wtd avg={vwa:+.2f} mm'
                    f'  (spd ext≈{np.abs(vel_a).mean():.3f}  ret≈{np.abs(vel_b).mean():.3f} cts/samp)'
                )

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


def make_normalized_plot(title, datasets, show_legend=True):
    """
    4th plot: subtract each dataset's overall mean so all curves are centred on
    zero, then report max absolute deviation from zero for positions > MIN_POSITION_MM.
    datasets: list of (label, x_mm, y_mm, color, legend_label, velocity)
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    summary_lines = []
    for label, x_mm, y_mm, color, leg, velocity in datasets:
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
        pos_mask = (x_mm >= MIN_POSITION_MM) & np.isfinite(y_norm)
        if pos_mask.sum() > 0:
            abs_dev  = np.abs(y_norm[pos_mask])
            max_dev  = abs_dev.max()
            worst_pos = x_mm[pos_mask][np.argmax(abs_dev)]
            worst_val = y_norm[pos_mask][np.argmax(abs_dev)]
            summary_lines.append(
                f'{leg}:  removed_mean={dataset_mean:+.2f}  '
                f'max_dev={max_dev:.2f} mm  at {worst_pos:.0f} mm ({worst_val:+.2f})'
            )

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Stringpot Position (mm)')
    ax.set_ylabel('Error − mean  (mm)')
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which='major', linestyle='-',  alpha=0.4)
    ax.grid(True, which='minor', linestyle=':', alpha=0.25)
    ax.set_ylim(-Y_SCALE_MM, Y_SCALE_MM)

    stats_text = '\n'.join(summary_lines)
    if show_legend:
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

        x_mm, y_mm, ext_mask, ret_mask, velocity = load_wdh(fp)

        for mask, tag in [(ret_mask, 'RET'), (ext_mask, 'EXT')]:
            vals = y_mm[mask]; ok = np.isfinite(vals)
            if ok.sum() == 0: continue
            mean = vals[ok].mean(); std = vals[ok].std()
            mn = vals[ok].min();    mx  = vals[ok].max(); n = ok.sum()
            spd = np.abs(velocity[mask][ok]).mean()
            print(f'{label} {tag}: mean={mean:+.2f} mm  std={std:.2f} mm  '
                  f'min={mn:+.2f}  max={mx:+.2f}  n={n}  avg_spd={spd:.4f} cts/samp')

        # Velocity-weighted average across both directions
        vwa = _vel_weighted_mean(y_mm[ext_mask], velocity[ext_mask],
                                 y_mm[ret_mask], velocity[ret_mask])
        if vwa is not None:
            spd_ext = np.abs(velocity[ext_mask]).mean()
            spd_ret = np.abs(velocity[ret_mask]).mean()
            print(f'{label} VEL-WTD-AVG: {vwa:+.2f} mm'
                  f'  (w_ext={1/spd_ext:.2f}  w_ret={1/spd_ret:.2f})')
        print()

        # velocity sliced to match each directional mask, passed through for
        # vel-weighted averaging inside make_plot
        all_ds.append((label, x_mm[ret_mask], y_mm[ret_mask], c_ret,
                       f'{label} — retract', velocity[ret_mask]))
        all_ds.append((label, x_mm[ext_mask], y_mm[ext_mask], c_ext,
                       f'{label} — extend',  velocity[ext_mask]))
        ret_ds.append((label, x_mm[ret_mask], y_mm[ret_mask], c_ret,
                       label, velocity[ret_mask]))
        ext_ds.append((label, x_mm[ext_mask], y_mm[ext_mask], c_ext,
                       label, velocity[ext_mask]))

    make_plot('All Data — Extend + Retract', all_ds)
    make_plot('Retract Only', ret_ds)
    make_plot('Extend Only',  ext_ds)
    make_normalized_plot('Mean-Normalised — All Data (max dev >35 mm shown)', all_ds)
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
