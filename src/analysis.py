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
    df = pd.read_csv(log_path)
    df.columns = ['time', 'x', 'y', 'z']

    # Calculate time range
    start = df.iloc[0]['time']
    end = df.iloc[-1]['time']
    duration = (end - start) / 1000000000  # nanoseconds to seconds

    n_batches = len(set(df['time']))
    duration *= (n_batches + 1) / n_batches  # Correct duration since end time is just the start of the last batch

    time_range = np.linspace(0, duration, len(df.index))

    # Plot data
    fig, ax = plt.subplots(1, 1)

    ax.plot(time_range, df['x'], label='x')
    ax.plot(time_range, df['y'], label='y')
    ax.plot(time_range, df['z'], label='z')

    ax.set_ylabel('Signal (V)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    fig.set_tight_layout(True)

    # Display or save file
    if save:
        plot_path = log_path.replace('.log', '.png')
        fig.savefig(plot_path, bbox_inches='tight')
        print(f"Saved plot to {plot_path}")
    else:
        fig.show()
