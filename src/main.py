from subsystems import Magnetometer
import time

device_name = 'Dev1'
mag = Magnetometer(device_name)

mag.start()
time.sleep(1)
mag.stop()

mag.close()


