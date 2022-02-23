import nidaqmx
import time
import queue
import threading

from typing import List

# Constants
DEVICE_NAME = "Dev2"
PHYSICAL_CHANNEL = DEVICE_NAME+ "/ai0"   # Physical channel name from NI-MAX
SAMPLE_RATE = 100               # DAQ sample rate in samples/sec
ACQ_DURATION = 5               # DAQ task duration in sec

LOG_FILE_PATH = "log.txt"

class DAQProducer:

    def __init__(self, device_name: str = None, channel: str = None, consumer_queues: List[queue.Queue] = None):
        if device_name is None:
            raise ValueError('DAQProduce requires a valid device name')

        if channel is None:
            raise ValueError('DAQProduce requires a valid channel')

        self.device = device_name
        self.channel = channel