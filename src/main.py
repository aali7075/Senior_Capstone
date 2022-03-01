import datetime
import time
import json
import os

from subsystems import Magnetometer, Panels, get_usgs


# set the directory of this path to be the cwd (current working directory)
# This is done so the task scheduler / cron job can use relative paths instead
# of absolute paths.
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def record_retrieve():
    acq_time = 5 * 60
    start = datetime.datetime.now()
    end = start + datetime.timedelta(seconds=acq_time)

    filename_date_format = '%Y-%m-%dT%H-%M-%S'
    start_file = start.strftime(filename_date_format)
    end_file = end.strftime(filename_date_format)

    device_name = 'cDAQ1Mod3'
    mag = Magnetometer(device_name)

    mag.start(f'readings/lab/mag_{start_file}__{end_file}.csv')
    time.sleep(acq_time)
    mag.stop()
    mag.close()

    print('Waiting a little bit for USGS data to catch up')
    time.sleep(5)

    print("Pinging USGS Boulder API")
    usgs_data = get_usgs(channels=['X', 'Y', 'Z'],
                         start_datetime=start,
                         end_datetime=end)

    usgs_path = f'../logs/usgs/usgs_{start_file}__{end_file}.json'
    with open(usgs_path, 'w') as fp:
        print('Successfully got USGS data, writing to file...')
        fp.write(json.dumps(usgs_data, indent=4))


def panels_example():
    device_name = 'Dev2'
    shape = [1, 1, 1]  # for a single panel
    panels = Panels(device_name, shape)

    # Play a 3 Hz 1.5 amp sin wave at 30 Hz for 10 seconds
    rate = 30  # Rate at which daq is updating
    amp = 1.5  # amp of sin wave
    freq = 3  # freq of sin wave
    sin = Panels.sin(rate, amp, freq)

    panels.start_loop(sin, rate)
    time.sleep(10)
    panels.stop()

    # Sets output to 0V, 1V, ..., 4V every second then stops
    q = panels.start_listening()
    for i in range(5):
        time.sleep(1)
        q.put_nowait(i)
    time.sleep(1)
    panels.stop()

    panels.close()


if __name__ == '__main__':
    record_retrieve()

    # panels_example()

    # filename = '../logs/lab_2-10-22.csv'
    # save = False
    # plot_log(filename, save)
    # plot_log_fft(filename, save, max_freq=60)
