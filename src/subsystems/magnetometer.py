from datetime import datetime

import numpy as np
import pandas as pd

import queue
import collections
import threading
import os

from time import perf_counter_ns, sleep

import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader


class Magnetometer:

    SCALING_FACTOR_mV_uT = 89.0 # mV/uT
    SCALING_FACTOR_T_V = SCALING_FACTOR_mV_uT * 1e-3

    def __init__(self, device_name, channel_names=None, sample_rate=50, buffer_size=50,
                 log_path='../logs/', average_buffer_size=10):
        """ Initializes magnetometer and prepares for reading data

        :param device_name: Name of device in NI MAX
        :param channel_names: Optional, list of channels to use. Default: ["/ai0", "/ai1", "/ai2"]
        :param sample_rate: Optional, sample rate in hz. Default: 200
        :param buffer_size: Optional, buffer size per producer loop. Default: 200
        """

        print("Initializing magnetometer...")
        self.buffer_size = buffer_size

        if not os.path.isdir(log_path):
            raise RuntimeError(f"Mag ERROR: Invalid log folder! '{log_path}' does not exist!")
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

        # Setup consumers
        self.log_queue = queue.Queue()
        self.average_queue = queue.Queue()
        self.consumers = [self.log_queue, self.average_queue]

        self.cons_man_thread = None
        self.log_thread = None
        self.average_thread = None

        self.average_buffer = collections.deque(maxlen=average_buffer_size)

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
            print("Mag WARNING: Already reading! Cannot start again")
            return

        # Setup threads
        if log_filename is None:
            log_filename = f"mag-{datetime.now().strftime('%m_%d_%y-%H_%M_%S')}.log"
        log_filename = self.log_path + log_filename
        self.log_thread = threading.Thread(target=self.__log_consumer, args=(log_filename, self.log_queue))
        self.average_thread = threading.Thread(target=self.__average_buffer_consumer, args=(self.average_queue,))
        self.cons_man_thread = threading.Thread(target=self.__consumer_manager,
                                                args=(self.daq_out_queue, self.consumers))

        # Start consumers
        self.log_thread.start()
        self.average_thread.start()
        self.cons_man_thread.start()

        # Start producer
        self.task.start()

        self.running = True
        print(f"Started magnetometer reading! Logging to {log_filename}")

        return log_filename

    def stop(self):
        """Stops magnetometer reading and logging, and sends poison pill to all consumers"""

        if not self.running:
            print("Mag WARNING: Magnetometer is not reading! Cannot stop")
            return

        print("Stopping magnetometer reading and logging...")

        # Stop producer
        self.task.stop()
        self.task.wait_until_done()
        self.daq_out_queue.put_nowait(None)

        # Wait for consumers to finish
        self.cons_man_thread.join()
        self.log_thread.join()
        self.average_thread.join()

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

    def __average_buffer_consumer(self, q):
        done = False
        while not done:
            data = q.get(block=True, timeout=2)

            if data is not None:
                timestamp = data[0]
                avgs = np.mean(data[2], axis=1)  # [z, y, x]
                avgs = np.flip(avgs)  # [x, y, z]

                self.average_buffer.append((timestamp, avgs))

            else:
                done = True

    def get_running_average(self, time_window):
        time_window *= (10 ** 9)
        window_start = perf_counter_ns() - time_window

        averages = None
        for timestamp, avgs in reversed(self.average_buffer):
            averages = np.concatenate((averages, [avgs]), axis=0) if averages is not None else [avgs]

            if timestamp < window_start:
                break

        return np.mean(averages, axis=1)

    def read_df(self, seconds=1):
        dfs = []

        def _read_to_df(self, task_idx, event_type, n_samples, callback_data=None):
            buffer = np.zeros((self.n_channels, n_samples), dtype=np.float64)
            n = self.reader.read_many_sample(buffer, n_samples)
            ts = perf_counter_ns()
            # self.daq_out_queue.put_nowait((ts, n, buffer))
            df = pd.DataFrame({'timestamp': ts, 'z': buffer[0], 'y': buffer[1], 'x': buffer[2]})
            dfs.append(df)
            return 0

        self.task.register_every_n_samples_acquired_into_buffer_event(self.buffer_size, _read_to_df)

        # Start producer
        self.task.start()
        print(f'starting data collection for {seconds} seconds')
        s = perf_counter_ns()
        while ((perf_counter_ns() - s) / 1e9) < 5:
            sleep(.01)
        print('finished data collection')
        self.close()

        df = pd.concat(dfs)
        df[['x', 'y', 'z']] *= 89.0 * 1e-3 # scaling factor to get tesla from volts
        return df
