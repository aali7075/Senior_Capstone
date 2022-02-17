import nidaqmx
import time
import queue
from multiprocessing import Process, Queue
import numpy as np
from time import perf_counter_ns
from nidaqmx.stream_readers import AnalogMultiChannelReader
from typing import List
from datetime import datetime, timedelta
import itertools
import requests
import json

# Constants
DEVICE_NAME = "cDAQ1Mod3"
X_CHANNEL = DEVICE_NAME + "/ai0"   # Physical channel name from NI-MAX
Y_CHANNEL = DEVICE_NAME + "/ai1"   # Physical channel name from NI-MAX
Z_CHANNEL = DEVICE_NAME + "/ai2"   # Physical channel name from NI-MAX

SAMPLE_RATE = 200               # DAQ sample rate in samples/sec
ACQ_DURATION = 5 * 60                # DAQ task duration in sec

def buffer_copy(in_q, out_qs: List[queue.Queue]):
    print("DAQ Copy Start")
    while True:
        data = in_q.get(block=True, timeout=2)
        if data is not None:
            for q in out_qs:
                q.put_nowait(data) # copy data to all queues
        else:
            print("DAQ Copy End")
            for q in out_qs:
                q.put_nowait(None) # send poison pill to all queues
            return

def log_data(q, log_path):
    print('Data logger start')
    all_data = ['timestamp, z, y, x']
    ref_time = None
    while True:
        data = q.get(block=True, timeout=2)
        if data is not None:

            if ref_time is None:
                ref_time = data[0]

            ts = [(data[0] - ref_time) * len(data[2][0])]
            zipped_data_chunk = map(lambda line: ', '.join(str(x) for x in line), zip(ts, *data[2]))
            all_data = itertools.chain(all_data, zipped_data_chunk)

            # z = data[2][0]
            # y = data[2][1]
            # x = data[2][2]

        else:
            print('Writing all collected data')
            with open(log_path, 'a') as fp:
                fp.write('\n'.join(all_data))
            print('Data logger end')
            return

# Main program
if __name__ == "__main__":
    num_channels = 3
    device = nidaqmx.system.Device(DEVICE_NAME)
    channels = device.ai_physical_chans.channel_names
    # print(channels)
    # before starting task, reset device if not shotdown correctly
    device.reset_device()

    # Set up DAQ vars
    task = nidaqmx.task.Task()
    task.ai_channels.add_ai_voltage_chan(X_CHANNEL)
    task.ai_channels.add_ai_voltage_chan(Y_CHANNEL)
    task.ai_channels.add_ai_voltage_chan(Z_CHANNEL)
    task.timing.cfg_samp_clk_timing(rate=SAMPLE_RATE,
                                    sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
    reader = AnalogMultiChannelReader(task.in_stream)

    nw = datetime.now()
    acq_end = nw + timedelta(seconds=int(ACQ_DURATION))
    filename_date_format = '%Y-%m-%dT%H-%M-%S'
    start_file = nw.strftime(filename_date_format)
    end_file = acq_end.strftime(filename_date_format)
    log_path = f"C:/Users/Lab/Documents/logs/lab_{start_file}__{end_file}.csv"

    # Set up threading vars
    daq_out_queue = Queue()
    log_copy_queue = Queue()

    # tuples with (queue, function, function args)
    copy_queues = [log_copy_queue,]
    worker_functions = [log_data,]
    worker_function_args = [[log_path],]

    # daq_worker = threading.Thread(target=daq_reader, args=(daq_out_queue, task))
    copy_worker = Process(target=buffer_copy, args=(daq_out_queue, copy_queues))

    workers = []
    for q, w_f, w_a in zip(copy_queues, worker_functions, worker_function_args):
        workers.append(Process(target=w_f, args=(q, *w_a)))

    def daq_reader(task_idx, event_type, num_samples, callback_data=None):
        buf = np.zeros((num_channels, num_samples), dtype=np.float64)
        n = reader.read_many_sample(buf, num_samples)
        ts = perf_counter_ns()
        daq_out_queue.put_nowait((ts, n, buf))
        return 0

    task.register_every_n_samples_acquired_into_buffer_event(200, daq_reader)

    print('Starting workers')
    copy_worker.start()
    for w in workers:
        w.start()

    time.sleep(1 * len(workers)) # give all the workers a little time to spool up
    print("All workers running.")

    # Start acquisition and threads
    task.start()
    print("Task running.")

    dur_ns = (ACQ_DURATION * 1e9)
    time_start = perf_counter_ns()
    t = perf_counter_ns()
    end_time = time_start + dur_ns
    i = 0
    while t < end_time:
        pct = int(((t - time_start) / dur_ns) * 100)
        if pct > i:
            print(f'{pct}% done.')
            i += 1
        t = perf_counter_ns()
        pass # Spin parent thread until task is done

    task.stop()
    daq_out_queue.put_nowait(None)
    print("Task is done.")

    print('waiting for workers to finish')
    while not daq_out_queue.empty():
        time.sleep(0.1)

    while not all(q.empty() for q in copy_queues):
        time.sleep(0.1)

    print('Joining workers.')
    copy_worker.join()
    for w in workers:
        w.join()

    print('Closing task.')
    task.close()

    print('Waiting 1 minute for USGS data to catch up')
    time.sleep(60)

    print("Pinging USGS Boulder API")
    api_date_format = '%Y-%m-%dT%H:%M:%S.000Z'
    start = nw.strftime(api_date_format)
    end = acq_end.strftime(api_date_format)
    params = {
        'elements': 'X,Y,Z',
        'endtime': end,
        'starttime': start,
        'format': 'json',
        'id': 'BOU',
        'sampling_period': 1,
        'type': 'adjusted'
    }
    api_url = f'https://geomag.usgs.gov/ws/data/'
    res = requests.get(api_url, params)

    usgs_path = f'C:/Users/Lab/Documents/logs/usgs_{start_file}__{end_file}.json'
    with open(usgs_path, 'w') as fp:
        if res.status_code == 200:
            print('Successfully got USGS data, writing to file...')
            fp.write(res.text)
        else:
            print('Failed to get USGS data...')
            fp.write('{"error": ' + res.status_code + '}')
