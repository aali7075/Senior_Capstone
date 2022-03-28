import datetime
import time
import json
import os

import numpy as np

from subsystems import Magnetometer, Panels, get_usgs, coil_diagnostics
from subsystems.analysis import resample_dataframe
from subsystems.constants import COIL_RESISTANCE, PCB_VOLTAGE_SCALING_FACTOR, MAGNETOMETER_SCALING_FACTOR
from simulation import FieldNuller

# set the directory of this path to be the cwd (current working directory)
# This is done so the task scheduler / cron job can use relative paths instead
# of absolute paths.
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def record_retrieve(seconds=1):
    acq_time = seconds * 60
    start = datetime.datetime.now()
    end = start + datetime.timedelta(seconds=acq_time)

    filename_date_format = '%Y-%m-%dT%H-%M-%S'
    start_file = start.strftime(filename_date_format)
    end_file = end.strftime(filename_date_format)

    device_name = 'cDAQ1Mod3'
    mag = Magnetometer(device_name)

    mag.start(f'readings/lab/mag_{start_file}__{end_file}.csv')
    time.sleep(acq_time)
    mag.stop()
    mag.close()

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


def cancel_fields():
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

    readings = None
    running = True

    hz = 1
    rate_period = 1.0/hz
    start_time = time.time()
    while running:
        readings = mag.get_running_average(rate_period)
        print(readings.shape)
        currents = nuller.solve(readings)
        print(currents)
        q.put_nowait(currents)

        # Loop rate control
        now = time.time() - start_time
        next_time = start_time + rate_period + now - (now % rate_period)
        while time.time() < next_time:
            pass


def run_diagnostics():
    mag_device = "cDAQ1Mod3"
    panels_device = "cDAQ1Mod4"

    logging_path = "../logs/temp/"
    mag = Magnetometer(mag_device, log_path=logging_path)

    shape = [2, 2, 1]
    panels = Panels(panels_device, shape)
    max_current = 50e-3  # amps
    daq_frequency = 60  # Hz
    sin_amplitude = max_current * COIL_RESISTANCE * PCB_VOLTAGE_SCALING_FACTOR  # V = IR
    target_frequency = 10  # Hz
    frequency_sweep = 1.5  # Hz
    sample_rate = 120  # Hz

    baseline = resample_dataframe(mag.read_df(seconds=5), sample_rate, trim=True)
    baseline[['x', 'y', 'z']] *= MAGNETOMETER_SCALING_FACTOR
    signal = panels.sin(daq_frequency, sin_amplitude, target_frequency)  # TODO: convert current to voltage
    num_coils = shape[0] * shape[1] * shape[2]
    for i in range(num_coils):
        all_signals = np.zeros(shape=(num_coils, len(signal)))
        all_signals[i] = signal
        panels.start_loop(all_signals, daq_frequency)
        panel_readings = resample_dataframe(mag.read_df(seconds=5), sample_rate, trim=True)
        panel_readings[['x', 'y', 'z']] *= MAGNETOMETER_SCALING_FACTOR
        panels.stop()
        freq, baseline_density, _, coil_density = coil_diagnostics(baseline,
                                                                   panel_readings,
                                                                   target_frequency,
                                                                   frequency_sweep,
                                                                   sample_rate)
        weights = frequency_sweep - np.abs(freq - frequency_sweep)
        weights /= np.sum(weights)
        impact = (coil_density - baseline_density) @ weights
        baseline_score = np.std(baseline_density @ weights)
        print(f'Baseline {baseline_score}')
        print(f'Panel {i} impact: {impact} -- Working={impact>baseline_score}')


def test_coils(voltage=5.0):
    panels_device = "cDAQ1Mod4"
    shape = [2, 2, 1]
    panels = Panels(panels_device, shape)
    applied_voltage = voltage * PCB_VOLTAGE_SCALING_FACTOR
    panels.start_loop(values=[applied_voltage] * 4, rate=1)


if __name__ == '__main__':
    cancel_fields()
    # record_retrieve()
    # panels_example()

    # filename = '../logs/lab_2-10-22.csv'
    # save = False
    # plot_log(filename, save)
    # plot_log_fft(filename, save, max_freq=60)
