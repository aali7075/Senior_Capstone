from subsystems import Magnetometer
import time
from analysis import plot_log

device_name = 'Dev1'
mag = Magnetometer(device_name)

log_file = mag.start()
time.sleep(10)
mag.stop()

mag.close()

plot_log(log_file)


