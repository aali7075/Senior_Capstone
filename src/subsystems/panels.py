import numpy as np
import threading
from queue import Queue

import nidaqmx
from nidaqmx.stream_writers import AnalogMultiChannelWriter
from nidaqmx.constants import AcquisitionType


class Panels:

    def __init__(self, device_name, shape, channels=None, default_rate=60):  # TODO: docstring
        # Input validation
        if len(shape) != 3:
            raise RuntimeError(f"Panels ERROR: Invalid shape for panels! {shape} should have 3 dimensions")
        if shape[0] not in [1, 2]:
            raise RuntimeError(f"Panels ERROR: Invalid number of panels! shape[0] should be 1 or 2")

        self.n_coils = np.prod(shape)

        if channels is not None:
            channels = np.array(channels)
            if channels.size != self.n_coils:
                channels = None
                print("Panels WARNING: Invalid channels shape! Using default numbering system")

        self.device_name = device_name
        self.shape = tuple(shape)

        self.default_rate = default_rate

        if channels is None:
            channels = [f"ao{i}" for i in range(self.n_coils)]
        self.channels = channels

        self.device = nidaqmx.system.Device(self.device_name)
        self.task = None

        self.thread = None
        self.running = False
        self.looping = False
        self.in_queue = Queue()

    def __del__(self):
        self.close()

    def close(self):
        """Deallocates resources"""
        if hasattr(self, 'task') and self.task is not None:
            self.task.close()
            self.task = None

    def __setup_task(self, rate=-1, samples=0):  # TODO: docstring
        self.close()

        self.device.reset_device()

        self.task = nidaqmx.task.Task(new_task_name='Coils Output')
        i = 0
        for panel in range(self.shape[0]):
            for x in range(self.shape[1]):
                for y in range(self.shape[2]):
                    channel_name = self.channels[i]
                    self.task.ao_channels.add_ao_voltage_chan(f'{self.device_name}/{channel_name}')

                    i += 1

        self.writer = AnalogMultiChannelWriter(self.task.out_stream)

        if rate > 0:
            self.task.timing.cfg_samp_clk_timing(rate,
                                                 sample_mode=AcquisitionType.CONTINUOUS,
                                                 samps_per_chan=samples)

    def start_loop(self, values, rate):  # TODO: docstring
        if self.running:
            print(f"Panels WARNING: Already {'looping' if self.looping else 'listening'}! Cannot start again")
            return

        self.thread = threading.Thread(target=self.__loop, args=(values,))
        self.__setup_task(rate, values.shape[1])

        self.running = True
        self.writer.write_many_sample(values)
        self.task.start()
        self.thread.start()

        self.looping = True
        print('Started panels looping...')

    def start_listening(self):  # TODO: docstring
        if self.running:
            print(f"Panels WARNING: Already {'looping' if self.looping else 'listening'}! Cannot start again")
            return

        self.thread = threading.Thread(target=self.__listen)
        self.__setup_task()

        self.running = True
        self.task.start()
        self.thread.start()

        self.looping = True
        print('Started panels listening...')

        return self.in_queue

    def stop(self):  # TODO: docstring
        if not self.running:
            print("Panels WARNING: Not reading! Cannot stop")
            return

        self.running = False

        self.thread.join()
        self.task.stop()
        self.task.wait_until_done()

        self.looping = False

        print("Stopped panels writing...")

    def __loop(self, values):  # TODO: docstring
        while self.running:
            pass
            # if self.task.out_stream.open_current_loop_chans_exist:
            #     print("Open loop!")

    def __listen(self):  # TODO: docstring
        initial_values = np.zeros(self.n_coils)
        self.in_queue.put_nowait(initial_values)

        while self.running:
            if not self.in_queue.empty():
                values = np.array(self.in_queue.get_nowait(), dtype=np.float64)
                if len(values) != self.n_coils:
                    print("Panels WARNING: Invalid shape in queue! length must be n_coils")
                    continue

                self.writer.write_one_sample(values)

    def index_to_name(self, index):  # TODO: docstring
        coils_per_panel = self.n_coils / self.shape[0]
        panel = 'A' if index < coils_per_panel else 'B'
        coil_index = index % coils_per_panel

        return f"{panel}{coil_index}"

    @staticmethod
    def sin(rate, amplitude, frequency, offset=0):
        """
        Generates values following sin wave for use in start_loop

        :param rate: Desired output rate (Hz). Should be significantly higher than freq
        :param amplitude: Desired amplitude of sin wave (V)
        :param frequency: Desired frequency of sin wave (Hz). Must evenly fit into rate
        :param offset: Optional offset (s)
        :return: values
        """

        if rate < frequency:
            print("Panels WARNING: rate cannot be lower than frequency")
            return None

        n_samples = float(rate) / frequency
        if n_samples % 1 != 0:
            print("Panels WARNING: frequency must fit into rate evenly")
            return None

        n_samples = int(n_samples)
        time = np.linspace(0, 1 / frequency, num=n_samples)
        values = amplitude * np.sin(2 * np.pi * frequency * (time - offset))

        return np.expand_dims(values, axis=0)
