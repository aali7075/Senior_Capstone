#########################################################################
# DAQmx Python - Producer/Consumer example
# Updated 10/19/2018
#
# Reads continuous samples from a single physical channel and writes
# them to a log file. Uses two parallel threads to perform
# DAQmx Read calls and data processing.
#
# Note: The number of samples per execution varies slightly since
# the task's start and stop times are specified in software.
#
# Input Type: Analog Voltage
# Dependencies: nidaqmx
#########################################################################

import nidaqmx
import time
import queue
import threading
import numpy as np
import matplotlib.pyplot as plt
from line_profiler_pycharm import profile
from time import perf_counter_ns
from nidaqmx.stream_readers import AnalogSingleChannelReader

# Constants
DEVICE_NAME = "cDAQ1Mod3"
PHYSICAL_CHANNEL = DEVICE_NAME + "/ai0"   # Physical channel name from NI-MAX
SAMPLE_RATE = 200               # DAQ sample rate in samples/sec
ACQ_DURATION = 10                # DAQ task duration in sec

LOG_FILE_PATH = "log.txt"

# Reads any available samples from the DAQ buffer and places them on the queue.
# Runs for ACQ_DURATION seconds.
@profile
def producer_loop(q, task):
    print("Producer start")
    reader = AnalogSingleChannelReader(task.in_stream)
    buffer = np.zeros((100000,))
    start_time = time.time()
    while time.time() - start_time < ACQ_DURATION:
        n = reader.read_many_sample(buffer)
        # print(len(data))
        # print(np.mean(data))
        q.put_nowait(np.mean(buffer[:n]))
    q.put_nowait(None)
    print("Producer end")
    task.close()
    return

# Takes samples from the queue and writes them to LOG_FILE_PATH.
def consumer_loop(q, buff, file):
    print("Consumer start")
    while True:
        temp = q.get(block=True, timeout=2)
        if temp is None:
            print("Consumer end")
            return
        #
        # for val in temp:
        file.write("{}\n".format(temp))
        buff.append(temp)

        # time.sleep(0.5) # Simulate 0.5 seconds of extra processing per sample
        q.task_done()

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

    out_file = open(LOG_FILE_PATH,"w+")
    out_file.write("\n~~New Test Started~~\n")

    # Set up threading vars
    q = queue.Queue()
    data_buffer = []
    prod = threading.Thread(target=producer_loop, args=(q, task))
    cons = threading.Thread(target=consumer_loop, args=(q, data_buffer, out_file))

    # Start acquisition and threads
    task.start()
    prod.start()
    cons.start()
    print("Task is running")

    while not task.is_task_done():
        pass # Spin parent thread until task is done
    print("Task is done")

    cons.join()
    print("Consumer finished")

    # Clean up
    out_file.close()
    task.close()

    print(len(data_buffer))

    # create time data
    t = np.arange(0, ACQ_DURATION, ACQ_DURATION/len(data_buffer))

    # plotting
    plt.plot(t, data_buffer, "-o")
    plt.title('NI DAQmx Voltage')
    plt.xlabel('t [s]')
    plt.ylabel('U [V]')
    plt.grid()
    plt.show()
    print("Done!")