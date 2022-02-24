import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime


def resample_dataframe(df, sample_rate, agg_func='mean', trim=False):
    """
    Aggregates a timeseries of observations into a timeseries of uniform sample rate.

    :param df: Dataframe of observations whose index is the elapsed seconds passed when the observation was recorded.
    :param sample_rate: Number of samples to aggregate to in each second. (Hz)
    :param agg_func: Function(s) to aggregate the sub-samples. Default is 'mean', can be a list of any agg functions.
    :param trim: Defaults to false. Choose to remove the final window from the dataframe if it is not fully saturated
                 with samples.
    :return: Re-sampled dataframe.

    Example, resampling to 240 Hz, removing excess data:
    >>> df
                       mean       min       max
    second sample
    0      0       1.891775  1.888062  1.895838
           1       1.890114  1.886317  1.895307
           2       1.891706  1.887824  1.895460
           3       1.893070  1.888302  1.896890
           4       1.892404  1.888322  1.895355
                     ...       ...       ...
    298    235     1.926796  1.921590  1.930197
           236     1.925259  1.921236  1.928926
           237     1.924941  1.920249  1.929355
           238     1.925252  1.921475  1.930207
           239     1.924471  1.921347  1.929047

    >>> resample_dataframe(df, sample_rate=120, agg_func=['min', 'mean', 'max'], trim=True)
                          z                      ...         x
                       mean       min       max  ...      mean       min       max
    second sample                                ...
    0      0       2.434054  2.432735  2.435856  ... -1.155265 -1.158869 -1.153813
           1       2.438351  2.436730  2.439159  ... -1.159620 -1.160807 -1.158771
           2       2.436643  2.434566  2.438289  ... -1.155815 -1.158811 -1.152495
           3       2.433168  2.432705  2.433764  ... -1.153587 -1.153922 -1.152583
           4       2.436046  2.434105  2.438172  ... -1.158350 -1.160781 -1.154522
                     ...       ...       ...  ...       ...       ...       ...
    298    235     2.432811  2.432433  2.433479  ... -1.110179 -1.110439 -1.109576
           236     2.435953  2.433901  2.438113  ... -1.114877 -1.117206 -1.110754
           237     2.437805  2.436395  2.438650  ... -1.114963 -1.115927 -1.112675
    """

    _df = df.copy(deep=True)
    _df['second'] = _df.index.astype(int)

    _df_list = []

    def _sub_sample(w_df):
        available_samples = len(w_df)
        if sample_rate >= available_samples:
            raise ValueError(f'Cannot aggregate to a higher sample rate than originally provided in current window.\n'
                             f'Samples in window: {available_samples}, desired sampling rate: {sample_rate}.')

        # Since each interval is 1 second, subtracting the min of the interval gives a parameterization for the sample
        # observation time in terms of the percentage of how far along it is in the window [0, 1).
        # Then multiple this percentage by the total samples in the window to get which re-sample it would fall under
        w_df['sample'] = ((w_df.index - w_df.index.min()) * sample_rate).astype(int)
        return w_df

    _df = _df.groupby('second').apply(_sub_sample)

    resampled_df = _df.groupby(by=['second', 'sample']).agg(agg_func)

    # if choosing to trim and the last window is not fully saturated
    if trim and len(resampled_df) != _df['second'].iloc[-1] * sample_rate:
        return resampled_df[:(len(resampled_df) // sample_rate) * sample_rate]
    else:
        return resampled_df


def usgs_dataframe(usgs_filepath):
    usgs_json = json.load(open(usgs_filepath, 'r'))
    times = [datetime.strptime(dt[:-5], '%Y-%m-%dT%H:%M:%S') for dt in usgs_json['times']]

    channels = {}
    for col in usgs_json['values']:
        channels[col['id']] = col['values']

    return pd.DataFrame(index=times, data=channels)


def merge_log_usgs(dt_identifier, sample_rate=240):
    log_path = f'../logs/readings/lab/mag_{dt_identifier}.csv'
    usgs_path = f'../logs/usgs/usgs_{dt_identifier}.json'
    rdf = resample_dataframe(read_log(log_path), sample_rate, trim=True)
    udf = usgs_dataframe(usgs_path)
    udf.columns = [f'USGS - {v}' for v in udf.columns]  # differentiate the column names

    # First, the data collected by the magnetometer needs a scaling factor applied to it
    # to get the data from V to T.
    # The scaling factor for the magnetometer is 89 mv/uT -> 8.9e-4 T/V
    rdf *= 8.9e-4

    # Next, scale the data collected form USGS to be in terms of T. Currently in nT
    udf *= 1e-9

    # Next, re-index the USGS data s.t. each row is elapsed seconds since the start
    udf.index = list(range(len(udf)))

    # trim excess seconds from USGS data
    udf = udf.iloc[:len(rdf.index.levels[0])]

    # join the two data frames on their `second` index (seconds elapsed)
    merged_df = rdf.reset_index()\
                   .set_index('second')\
                   .join(udf, how='left')\
                   .set_index('sample', append=True)  # re-index on the sample #

    merged_df.index.names = ['second', 'sample']  # fix the index name after the merge

    return merged_df


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


def fft_signal(signal, sampling_rate):
    sig = signal - np.mean(signal)
    freq = np.fft.rfftfreq(len(sig), d=1.0/sampling_rate)
    fft = np.abs(np.fft.rfft(sig)) ** 2  # a density...
    return freq, fft


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
        if peak[0] > 0.001:
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
        ax.annotate("(%.2f, %f)" % peak, xy=peak, textcoords='data')
    ax.set_ylabel('Intensity')
    ax.set_xlabel('Frequency (Hz)')
    ax.legend()
    fig.set_tight_layout(True)

    # Display or save file
    if save:
        plot_path = log_path.replace('.csv', '_fft.png')
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

    return df.set_index('time')
