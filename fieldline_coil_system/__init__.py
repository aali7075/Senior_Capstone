from datetime import datetime, timedelta
import time
import json
import os
from typing import Union
from time import perf_counter_ns

import numpy as np

from .subsystems import analysis
from .subsystems import Magnetometer, Panels, get_usgs, coil_diagnostics
from .subsystems.analysis import *
from .subsystems.constants import RESISTANCE, PCB_INVERSION, MAGNETOMETER_SCALING_FACTOR
from .simulation import FieldNuller

# set the directory of this path to be the cwd (current working directory)
# This is done so the task scheduler / cron job can use relative paths instead
# of absolute paths.
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def record_retrieve(seconds: Union[int, float] = 1.0, usgs: bool = False, plot: bool = False):
    """
    Record magnetometer and optional usgs data. Saves raw recorded data to a file.
    :param seconds: Number of seconds to record magnetometer data.
    :param usgs: Enable fetching of concurrent USGS Boulder data.
    :return: None
    """

    if type(seconds) not in (int, float) or seconds <= 0:
        raise ValueError(f'seconds must be a positive float or integer.')

    if not isinstance(usgs, bool):
        raise ValueError(f'usgs flag must be a bool.')

    start = datetime.now()
    end = start + timedelta(seconds)

    filename_date_format = '%Y-%m-%dT%H-%M-%S'
    start_file = start.strftime(filename_date_format)
    end_file = end.strftime(filename_date_format)

    device_name = 'cDAQ1Mod3'
    mag = Magnetometer(device_name)

    path_name = f'readings/lab/mag_{start_file}__{end_file}.csv'
    log_file = mag.start(path_name)
    time.sleep(seconds)
    mag.stop()
    mag.close()

    print(f'Magnetometer readings saved to: {path_name}')
    if plot:
        plot_log(log_file)

    if usgs:
        print('Waiting a little bit for USGS data to catch up')
        time.sleep(5)

        print("Pinging USGS Boulder API")
        usgs_data = get_usgs(channels=['X', 'Y', 'Z'],
                             start_datetime=start,
                             end_datetime=end)

        usgs_path = f'../logs/usgs/usgs_{start_file}__{end_file}.json'
        with open(usgs_path, 'w') as fp:
            print('Successfully got USGS data, writing to file...')
            fp.write(json.dumps(usgs_data, indent=4))
            print(f'USGS readings saved to: {usgs_path}')


def cancel_fields(hz: int = 10, verbose: bool = True):
    """
    Run the field cancellation control loop at the specified frequency.
    :param hz: Frequency to update the control loop.
    :param verbose: Print debug info when running.
    :return: None
    """

    if not isinstance(hz, int):
        raise ValueError('hz parameter must be an integer.')

    if not isinstance(verbose, bool):
        raise ValueError('verbose flag must be a bool.')

    mag_device = "cDAQ1Mod3"
    panels_device = "cDAQ1Mod4"

    logging_path = "../logs/temp/"
    mag = Magnetometer(mag_device, log_path=logging_path)

    shape = [2, 2, 1]
    panels = Panels(panels_device, shape)

    coil_size = (.040, .040)  # meters
    coil_spacing = 0.00254  # meters
    wall_spacing = 0.047  # meters
    max_current = 1  # amps
    nuller = FieldNuller(shape, coil_size, coil_spacing, wall_spacing, max_current)

    measurement_point = (0, 0, 0)
    nuller.set_point(measurement_point)
    mag.start()
    q = panels.start_listening()
    running = True

    rate_period = 1.0/hz
    start_time = time.time()
    while running:
        readings = mag.get_running_average(rate_period)
        if verbose:
            print(f'Readings: {readings}')
        currents = nuller.solve(readings)

        # applied voltage = given current * coil resistance
        # applied voltage = (0 - desired voltage) * (coil resistance / 200 Ohms, R-sense)
        # find desired voltage:
        # voltage = -200.0 *

        voltages = currents * RESISTANCE * PCB_INVERSION  # element-wise multiplication
        if verbose:
            print(f'Solved voltages: {voltages}')
        q.put_nowait(voltages)

        # Loop rate control
        now = time.time() - start_time
        if now > 5:
            running = False
        next_time = start_time + rate_period + now - (now % rate_period)
        while time.time() < next_time:
            pass

    mag.stop()
    panels.stop()

    mag.close()
    panels.close()


def run_diagnostics():
    """
    Run the coil diagnostics test.
    :return: None
    """
    mag_device = "cDAQ1Mod3"
    panels_device = "cDAQ1Mod4"
    logging_path = "../logs/temp/"

    shape = [2, 2, 1]
    panels = Panels(panels_device, shape)
    max_current = 50e-3  # amps
    daq_frequency = 60  # Hz
    target_frequency = 10  # Hz
    frequency_sweep = 1.5  # Hz
    sample_rate = 120  # Hz
    panel_start_times = []
    panel_end_times = []

    with Magnetometer(mag_device, log_path=logging_path) as mag:
        signal_len = len(panels.sin(daq_frequency, 0, target_frequency)[0])
        num_coils = shape[0] * shape[1] * shape[2]

        s = perf_counter_ns()
        log_path = mag.start()
        time.sleep(5)

        zero_signal = np.zeros(shape=(num_coils, signal_len))

        for i in range(num_coils):
            print(f'Testing coil {i}')
            all_signals = np.zeros(shape=(num_coils, signal_len))
            sin_amplitude = max_current * RESISTANCE * PCB_INVERSION[i]  # V = IR
            all_signals[i] = panels.sin(daq_frequency, sin_amplitude, target_frequency)[0]

            if panels.running:
                panel_end_times.append((perf_counter_ns() - s) / 1e9)
                panels.update_loop(all_signals)
            else:
                panels.start_loop(all_signals, daq_frequency)
            panel_start_times.append((perf_counter_ns() - s) / 1e9)

            time.sleep(5)

        panel_end_times.append((perf_counter_ns() - s) / 1e9)
        panels.update_loop(zero_signal)
        mag.stop()
        mag.close()

    print('Loading collected diagnostic data')
    all_data = resample_dataframe(read_log(log_path), sample_rate, trim=True)
    baseline = all_data.loc[0:int(panel_start_times[0]) - 1]
    for i in range(num_coils):
        print(f'Analyzing coil {i}')
        panel_readings = all_data.loc[int(panel_start_times[i]):int(panel_end_times[i]) - 1]
        freq, baseline_density, _, coil_density = coil_diagnostics(baseline,
                                                                   panel_readings,
                                                                   target_frequency,
                                                                   frequency_sweep,
                                                                   sample_rate)

        weights = frequency_sweep - np.abs(freq - frequency_sweep)
        weights /= np.sum(weights)
        impact = (coil_density - baseline_density) @ weights
        baseline_score = np.std(baseline_density @ weights)

        print(f'\tBaseline {baseline_score}')
        print(f'\tPanel {i} impact: {impact} -- Working={impact>baseline_score}')
    plot_log_fft(log_path)


def test_coils(current: Union[int, float] = 5.0, panels=None):
    """
    Apply the same current to all coils.
    :param current:
    :return:
    """

    if type(current) not in (int, float):
        raise ValueError('current must be a float.')

    shape = [2, 2, 1]
    if panels is None:
        panels_device = "cDAQ1Mod4"
        panels = Panels(panels_device, shape)
    else:
        if not isinstance(panels, Panels):
            raise ValueError('panels must be a Panels object')

    applied_voltage = np.ones(np.prod(shape)) * current * RESISTANCE * PCB_INVERSION
    print(applied_voltage)
    if not panels.running:
        print('Applying voltage to coils...')
        panels.start_listening()
        panels.in_queue.put_nowait(applied_voltage)
    else:
        print('Updating voltage to coils...')
        panels.in_queue.put_nowait(applied_voltage)

    return panels


def test_coils_ac(current: Union[int, float] = 5.0, hz=10, panels=None):
    """
    Apply the same current to all coils.
    :param current:
    :return:
    """

    if type(current) not in (int, float):
        raise ValueError('current must be a float.')

    shape = [2, 2, 1]
    if panels is None:
        panels_device = "cDAQ1Mod4"
        panels = Panels(panels_device, shape)
    else:
        if not isinstance(panels, Panels):
            raise ValueError('panels must be a Panels object')

    sin_amplitude = current * RESISTANCE * PCB_INVERSION
    signals = np.array([panels.sin(60, amp, hz)[0] for amp in sin_amplitude])
    if not panels.running:
        print('Applying voltage to coils...')
        panels.start_loop(signals, 60)
    else:
        print('Updating voltage to coils...')
        panels.update_loop(signals)

    return panels


if __name__ == '__main__':
    cancel_fields()
