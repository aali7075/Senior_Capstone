from datetime import datetime

import numpy as np
import pandas as pd

import queue
import threading
import os

from time import perf_counter_ns

import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader


class Magnetometer:
    def __init__(self, device_name, channel_names=None, sample_rate=200, buffer_size=200, log_path='../logs/'):
        """ Initializes magnetometer and prepares for reading data

        :param device_name: Name of device in NI MAX
        :param channel_names: Optional, list of channels to use. Default: ["/ai0", "/ai1", "/ai2"]
        :param sample_rate: Optional, sample rate in hz. Default: 200
        :param buffer_size: Optional, buffer size per producer loop. Default: 200
        """

        print("Initializing magnetometer...")

        if not os.path.isdir(log_path):
            raise OSError(f"Invalid log folder! '{log_path}' does not exist!")
        self.log_path = log_path

        self.device_name = device_name
        self.channels = channel_names if channel_names is not None else ["/ai0", "/ai1", "/ai2"]
        self.n_channels = len(self.channels)
        self.running = False

        # Reset device
        device = nidaqmx.system.Device(self.device_name)
        device.reset_device()

        # Setup task
        self.task = nidaqmx.task.Task(new_task_name='Magnetometer Input')
        for channel in self.channels:
            self.task.ai_channels.add_ai_voltage_chan(self.device_name + channel)

        self.task.timing.cfg_samp_clk_timing(rate=sample_rate,
                                             sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
        self.reader = AnalogMultiChannelReader(self.task.in_stream)

        # Setup producer
        self.daq_out_queue = queue.Queue()
        self.task.register_every_n_samples_acquired_into_buffer_event(buffer_size, self.__daq_producer)

        # Setup consumer
        self.log_queue = queue.Queue()
        self.consumers = [self.log_queue]

        self.cons_man_thread = None
        self.log_thread = None

        print("Magnetometer ready to start reading!")

    def __del__(self):
        """

        :return: test
        """
        self.close()

    def close(self):
        """Deallocates resources"""
        if hasattr(self, 'task') and self.task is not None:
            self.task.close()
            self.task = None

    def start(self, log_filename=None):
        """ Starts reading, logging, and passing data to any additional consumers

        :param log_filename: Optional name of log file
        :return: Path to log file
        """

        if self.running:
            print("WARNING: Magnetometer is already reading! Cannot start again")
            return

        # Setup threads
        if log_filename is None:
            log_filename = f"mag-{datetime.now().strftime('%m_%d_%y-%H_%M_%S')}.log"
        log_filename = self.log_path + log_filename
        self.log_thread = threading.Thread(target=self.__log_consumer, args=(log_filename, self.log_queue))
        self.cons_man_thread = threading.Thread(target=self.__consumer_manager,
                                                args=(self.daq_out_queue, self.consumers))

        # Start consumers
        self.log_thread.start()
        self.cons_man_thread.start()

        # Start producer
        self.task.start()

        self.running = True
        print(f"Started magnetometer reading! Logging to {log_filename}")

        return log_filename

    def stop(self):
        """Stops magnetometer reading and logging, and sends poison pill to all consumers"""

        if not self.running:
            print("WARNING: Magnetometer is not reading! Cannot stop")
            return

        print("Stopping magnetometer reading and logging...")

        # Stop producer
        self.task.stop()
        self.task.wait_until_done()
        self.daq_out_queue.put_nowait(None)

        # Wait for consumers to finish
        self.cons_man_thread.join()
        self.log_thread.join()

        self.running = False
        print("Magnetometer stopped!")

    def add_consumer(self, q):
        """ Adds a queue to be populated by the producer

        :param q: Queue to be populated
        """

        self.consumers.append(q)

    def __daq_producer(self, task_idx, event_type, n_samples, callback_data=None):
        buffer = np.zeros((self.n_channels, n_samples), dtype=np.float64)
        n = self.reader.read_many_sample(buffer, n_samples)
        ts = perf_counter_ns()
        self.daq_out_queue.put_nowait((ts, n, buffer))
        return 0

    @staticmethod
    def __consumer_manager(prod_q, cons_qs):
        done = False
        while not done:
            data = prod_q.get(block=True, timeout=2)

            if data is not None:
                # Copy data to all queues
                for q in cons_qs:
                    q.put_nowait(data)
            else:
                # Send poison pill to all queues and exit loop
                for q in cons_qs:
                    q.put_nowait(None)
                done = True

    @staticmethod
    def __log_consumer(filepath, log_queue):
        done = False
        while not done:
            data = log_queue.get(block=True, timeout=2)

            if data is not None:
                # Format and log data
                df = pd.DataFrame({'timestamp': data[0], 'z': data[2][0], 'y': data[2][1], 'x': data[2][2]})
                df.to_csv(filepath, mode='a', index=False, header=False)
            else:
                # Poison pill, exit loop
                done = True
