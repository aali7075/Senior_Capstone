import nidaqmx
import time
import queue
import threading
import numpy as np
from time import perf_counter_ns
from nidaqmx.stream_readers import AnalogSingleChannelReader
from typing import List
import pandas as pd

# Constants
DEVICE_NAME = "cDAQ1Mod3"
PHYSICAL_CHANNEL = DEVICE_NAME + "/ai0"   # Physical channel name from NI-MAX
SAMPLE_RATE = 200               # DAQ sample rate in samples/sec
ACQ_DURATION = .1 * 60                # DAQ task duration in sec

# Reads any available samples from the DAQ buffer and places them on the queue.
# Runs for ACQ_DURATION seconds.
def daq_reader(q, task):
    print("DAQ Reader start")
    reader = AnalogSingleChannelReader(task.in_stream)
    buffer = np.zeros((10,))
    start_time = time.time()
    while time.time() - start_time < ACQ_DURATION:
        n = reader.read_many_sample(buffer, number_of_samples_per_channel=10)
        ts = perf_counter_ns()
        q.put_nowait((ts, n, buffer[:n]))
    q.put_nowait(None)
    print("DAQ Reader Ennd")
    task.close()
    return

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

def log_data(q, fp):
    print('Data logger start')
    while True:
        data = q.get(block=True, timeout=2)
        if data is not None:
            print(data)
            # df = pd.DataFrame(columns=['timestamp', 'sample'],
            #                   data=[[data[0]] * data[1], data[2]])
            # df.to_csv(fp, mode='a')
        else:
            print('Data logger end')
            return


# Main program
if __name__ == "__main__":

    device = nidaqmx.system.Device(DEVICE_NAME)
    channels = device.ai_physical_chans.channel_names
    # print(channels)
    # before starting task, reset device if not shotdown correctly
    device.reset_device()

    # Set up DAQ vars
    task = nidaqmx.task.Task()
    task.ai_channels.add_ai_voltage_chan(PHYSICAL_CHANNEL)
    task.timing.cfg_samp_clk_timing(rate=SAMPLE_RATE,
                                    sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

    LOG_FILE_PATH = "lab_logs.csv"

    # Set up threading vars
    daq_out_queue = queue.Queue()
    log_copy_queue = queue.Queue()

    # tuples with (queue, function, function args)
    copy_queues = [log_copy_queue,]
    worker_functions = [log_data,]
    worker_function_args = [[LOG_FILE_PATH],]

    daq_worker = threading.Thread(target=daq_reader, args=(daq_out_queue, task))
    copy_worker = threading.Thread(target=buffer_copy, args=(daq_out_queue, copy_queues))

    workers = []
    for q, w_f, w_a in zip(copy_queues, worker_functions, worker_function_args):
        workers.append(threading.Thread(target=w_f, args=(q, *w_a)))

    # Start acquisition and threads
    task.start()
    print("Task running.")

    daq_worker.start()
    copy_worker.start()
    for w in workers:
        w.start()
    print("All workers running.")

    while not task.is_task_done():
        pass # Spin parent thread until task is done
    print("Task is done.")

    print('Joining workers.')
    daq_worker.join()
    copy_worker.join()
    for w in workers:
        w.join()

    print('Closing task.')
    task.close()