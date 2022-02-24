import datetime
import json

from subsystems import Magnetometer, get_usgs
import time
import os

# set the directory of this path to be the cwd (current working directory)
# This is done so the task scheduler / cron job can use relative paths instead
# of absolute paths.
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def record_retrieve():
    acq_time = 1 * 60
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


if __name__ == '__main__':
    record_retrieve()
