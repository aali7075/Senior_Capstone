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
from .subsystems.constants import RESISTANCE, PCB_INVERSION, MAGNETOMETER_SCALING_FACTOR, directory
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

    path_name = f'readings/mag_{start_file}__{end_file}.csv'
    log_file = mag.start(path_name)
    time.sleep(seconds)
    mag.stop()
    mag.close()

    print(f'Magnetometer readings saved to: {path_name}')
    if plot:
        plot_log(log_file,tesla=True)

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

    coil_size = (.2, .2)  # meters
    coil_spacing = 0.003  # meters
    wall_spacing = 0.2315  # meters
    turns_per_coil = 100
    max_current = 45e-3  # amps
    measurement_point = (0, 0.0406, 0)
    nuller = FieldNuller(shape=shape,
                         coil_size=coil_size,
                         coil_spacing=coil_spacing,
                         wall_spacing=wall_spacing,
                         turns_per_coil=turns_per_coil,
                         max_current=max_current,
                         point=measurement_point)

    log_path = mag.start()
    q = panels.start_listening()
    running = True

    rate_period = 1.0/hz
    start_time = time.time()

    panel_solutions = []

    while running:
        readings = mag.get_running_average(rate_period)
        if verbose:
            print(f'Readings: {readings}')

        currents = nuller.solve(readings)
        voltages = currents * RESISTANCE * PCB_INVERSION  # element-wise multiplication

        if verbose:
            print(f'Solved voltages: {voltages}')
        q.put_nowait(voltages)

        cv = {str(i): currents[i] for i in range(4)}
        panel_solutions.append({**{'timestamp': time.time()}, **cv})

        # Loop rate control
        now = time.time() - start_time
        if now > 5:
            running = False
        next_time = start_time + rate_period + now - (now % rate_period)
        while time.time() < next_time:
            pass

    panel_log_path = '../logs/panel/cancel.log'
    pd.DataFrame(panel_solutions).to_csv(panel_log_path, index=False)

    print('Stopping and closing')
    mag.stop()
    panels.stop()
    mag.close()
    panels.close()

    print('')
    # plot_log(log_path)


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
    daq_frequency = 120  # Hz
    target_frequency = 10  # Hz
    frequency_sweep = 1.5  # Hz
    sample_rate = 240  # Hz
    panel_start_times = []
    panel_end_times = []

    read_time = 5
    pause_time = 2

    with Magnetometer(mag_device, log_path=logging_path) as mag:
        signal_len = len(panels.sin(daq_frequency, 0, target_frequency)[0])
        num_coils = shape[0] * shape[1] * shape[2]

        s = perf_counter_ns()
        log_path = mag.start()
        time.sleep(read_time)

        zero_signal = np.zeros(shape=(num_coils, signal_len))

        for i in range(num_coils):
            print(f'Testing coil {i}')
            all_signals = np.zeros(shape=(num_coils, signal_len))
            sin_amplitude = max_current * RESISTANCE * PCB_INVERSION[i]  # V = IR
            all_signals[i] = panels.sin(daq_frequency, sin_amplitude, target_frequency)[0]

            if panels.running:
                panel_end_times.append((perf_counter_ns() - s) / 1e9)
                panels.stop()

                # wait between panels to minimize bleeding
                panels.start_loop(all_signals * 0, daq_frequency)
                time.sleep(pause_time)
                panels.stop()

                panels.start_loop(all_signals, daq_frequency)
            else:
                panels.start_loop(all_signals, daq_frequency)
            panel_start_times.append((perf_counter_ns() - s) / 1e9)

            time.sleep(read_time)

        panel_end_times.append((perf_counter_ns() - s) / 1e9)
        panels.stop()
        panels.start_loop(zero_signal, daq_frequency)

        print('mag stop')
        mag.stop()

        print('panel stop')
        panels.stop()

        print('mag close')
        mag.close()

        print('panels close')
        panels.close()

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
    # plot_log(log_path)
    # plot_log_fft(log_path)
    # plot_log_fft_multiple(log_path, list(sorted((panel_start_times + panel_end_times))))


def test_coils(current: Union[int, float] = 25e-3, panels=None):
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


def test_coils_ac(current: Union[int, float] = 25e-3, hz=10, panels=None):
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


def debug_test(coil_idx=0):
    coil_idx = int(coil_idx)
    #directory()
    mag_device = "cDAQ1Mod3"
    panels_device = "cDAQ1Mod4"
    logging_path = "../logs/temp/"

    current = 25e-3  # amps
    sample_rate = 120  # Hz
    rate_period = 1.0/10.0

    shape = [2, 2, 1]
    panels = Panels(panels_device, shape)

    coil_size = (.2, .2)  # meters
    coil_spacing = 0.003  # meters
    # coil_spacing = 0.000  # meters
    wall_spacing = 0.2315  # meters
    # wall_spacing = 0.0  # meters
    max_current = 45e-3  # amps
    turns_per_coil = 100
    measurement_point = (0, 0.0406, 0)

    nuller = FieldNuller(shape=shape,
                         coil_size=coil_size,
                         coil_spacing=coil_spacing,
                         wall_spacing=wall_spacing,
                         turns_per_coil=turns_per_coil,
                         max_current=max_current,
                         point=measurement_point)

    q = panels.start_listening()
    applied_voltage = np.zeros(shape=(4,))
    q.put_nowait(applied_voltage)

    # use to adjust the orientation of the magnetometer such that the axis of it's x,y,z readings align with the
    # x,y,z of the simulation.
    orientation = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
    with Magnetometer(mag_device, log_path=logging_path, orientation=orientation) as mag:
        log_path = mag.start()

        # read ambient field for 5 seconds
        print('reading ambient field')
        time.sleep(5)

        # The field components we expect to generate at the magnetometer with coil 0
        # at the pre-determined current (Tesla)
        currents = np.zeros(shape=(np.prod(shape),))
        currents[coil_idx] = current
        expected_field_contribution = nuller.b_mat @ currents

        # get running average over past 1/10 of a second (Tesla)
        readings = mag.get_running_average(0.1)

        # How we expect the system to look w/ our contribution (Tesla)
        expected_agg_field = expected_field_contribution + readings

        # convert applied currents to voltages and send to DAQ
        print('Creating field')
        voltages = currents * RESISTANCE * PCB_INVERSION
        q.put_nowait(voltages)
        time.sleep(5)

        updated_readings = mag.get_running_average(0.1)

    print('Stopping and closing')
    mag.stop()
    panels.stop()
    time.sleep(5)
    mag.close()
    panels.close()

    print('Initial Readings:\t', readings)
    print('')

    print('Expected Post Readings:\t', expected_agg_field)
    print('Actual Post Readings:\t', updated_readings)
    # print(f'L2 error: {np.linalg.norm(updated_readings - expected_agg_field)}')
    print('')

    actual_field_contribution = updated_readings - readings
    relative_error = (actual_field_contribution - expected_field_contribution) / expected_field_contribution

    print(f'Coil {coil_idx} Expected Contribution:\t', expected_field_contribution)
    print(f'Coil {coil_idx} Actual Contribution:\t', actual_field_contribution)
    print(f'Relative X error: {relative_error[0]}')
    print(f'\t- Correct X direction: {np.sign(relative_error[0])}')
    print(f'Relative Y error: {relative_error[1]}')
    print(f'\t- Correct Y direction: {np.sign(relative_error[1])}')
    print(f'Relative Z error: {relative_error[2]}')
    print(f'\t- Correct Z direction: {np.sign(relative_error[2])}')
    print('')

    # Read in and normalize magnetometer readings over the experiment (Volts)
    mag_df = resample_dataframe(read_log(log_path), sample_rate, trim=True)[['x', 'y', 'z']]

    # Convert readings from volts to Tesla
    mag_df *= MAGNETOMETER_SCALING_FACTOR

    mag_df[['x', 'y', 'z']].plot()
    plt.axhline(y=expected_agg_field[0], color='blue', linestyle='-.', label=f'Expected x')
    plt.axhline(y=expected_agg_field[1], color='orange', linestyle='-.', label=f'Expected y')
    plt.axhline(y=expected_agg_field[2], color='green', linestyle='-.', label=f'Expected z')

    # plt.axhline(y=readings[0], color='r', linestyle='-.', label=f'rx')
    # plt.axhline(y=readings[1], color='g', linestyle='-.', label=f'ry')
    # plt.axhline(y=readings[2], color='b', linestyle='-.', label=f'rz')

    plt.ylabel('Tesla')
    plt.xlabel('Time (second, sample)')
    plt.title(f'Coil {coil_idx} Test - {current * 1000:.2f}mA')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    debug_test()
