from subsystems import Magnetometer
import time
from analysis import plot_log, plot_log_fft

device_name = 'Dev1'
mag = Magnetometer(device_name)

log_file = mag.start()
time.sleep(10)
mag.stop()

mag.close()

plot_log(log_file)
plot_log_fft(log_file, max_freq=1)
