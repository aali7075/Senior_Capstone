import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_log(log_path, save=False):
    """
    Reads data from log file and plots the data

    :param log_path: Path to log file
    :param save: Whether to save or display plot. Default: False (display)
    """
    # Read data from log file
    df = read_log(log_path)

    # Plot data
    fig, ax = plt.subplots(1, 1)

    ax.plot(df['time'], df['x'], label='x')
    ax.plot(df['time'], df['y'], label='y')
    ax.plot(df['time'], df['z'], label='z')

    ax.set_ylabel('Signal (V)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    fig.set_tight_layout(True)

    # Display or save file
    if save:
        plot_path = log_path.replace('.csv', '.png')
        fig.savefig(plot_path, bbox_inches='tight')
        print(f"Saved plot to {plot_path}")
    else:
        fig.show()


def plot_log_fft(log_path, save=False, max_freq=60):
    """
    Reads data from log file, computes and plots fft

    :param log_path: Path to log file
    :param save: Whether to save or display plot. Default: False (display)
    :param max_freq: Show up to this frequency on plot (Hz).
    """

    # Read data from log
    df = read_log(log_path)
    length = len(df.index)
    spacing = df['time'][1]

    # Calculate frequency space and trim to max_freq
    freq = np.fft.rfftfreq(len(df.index), d=spacing)
    max_index = np.argmax(freq > max_freq)
    freq = freq[:max_index]

    # Init plotting variables
    fig, ax = plt.subplots(1, 1)
    peaks = []

    for i in ['x', 'y', 'z']:
        # Calculate and trim FFT
        fft = np.abs(np.fft.rfft(df[i])) / length
        fft = fft[:max_index]

        # Plot FFT
        ax.plot(freq, fft, label=i)

        # Calculate peak for this dimension
        peak_index = np.argmax(fft)
        peak = (freq[peak_index], fft[peak_index])

        # Only label peak if it is far enough away from previous peaks
        new_peak = True
        threshold = .1  # As a percentage of the graph width
        for p_x, _ in peaks:
            if abs(p_x - peak[0]) < threshold * max_freq:
                new_peak = False
                break
        if new_peak:
            peaks.append(peak)

    # Format plot
    for peak in peaks:
        ax.annotate("(%.2f, %.2f)" % peak, xy=peak, textcoords='data')
    ax.set_ylabel('Intensity')
    ax.set_xlabel('Frequency (Hz)')
    ax.legend()
    fig.set_tight_layout(True)

    # Display or save file
    if save:
        plot_path = log_path.replace('.log', '_fft.png')
        fig.savefig(plot_path, bbox_inches='tight')
        print(f"Saved plot to {plot_path}")
    else:
        fig.show()


def read_log(log_path):
    """
    Reads data from log into dataframe and transforms time into seconds

    :param log_path: Path to log file
    """
    # Read data from log file
    df = pd.read_csv(log_path)
    df.columns = ['time', 'z', 'y', 'x']
    df['time'] = df['time'] - min(df['time'])  # zero out times

    prev_df = None
    start_ts = None
    fixed_dfs = []
    times = []
    for ts, wdf in df.groupby('time'):
        if prev_df is None:
            prev_df = wdf.copy(deep=True)
            start_ts = ts
            continue

        prev_df['time'] = np.linspace(start_ts, ts, num=len(prev_df))
        fixed_dfs.append(prev_df)
        prev_df = wdf.copy(deep=True)
        times.append(ts - start_ts)
        start_ts = ts

    ts = start_ts + np.mean(times)
    prev_df['time'] = np.linspace(start_ts, ts, num=len(prev_df))
    fixed_dfs.append(prev_df)

    df = pd.concat(fixed_dfs)
    df['time'] /= 1_000_000_000  # ns -> s
    df = df.set_index('time')

    return df
