import datetime
import requests
from subsystems import Magnetometer
import time
from analysis import plot_log, plot_log_fft

acq_time = .1 * 60
start = datetime.datetime.now()
end = start + datetime.timedelta(seconds=acq_time)

filename_date_format = '%Y-%m-%dT%H-%M-%S'
start_file = start.strftime(filename_date_format)
end_file = end.strftime(filename_date_format)

device_name = 'cDAQ1Mod3'
mag = Magnetometer(device_name)

log_file = mag.start(f'readings/lab/mag_{start_file}__{end_file}.csv')
time.sleep(acq_time)
mag.stop()
mag.close()

print('Waiting a little bit for USGS data to catch up')
time.sleep(5)

print("Pinging USGS Boulder API")
api_date_format = '%Y-%m-%dT%H:%M:%S.000Z'
params = {
    'elements': 'X,Y,Z',
    'endtime': end.strftime(api_date_format),
    'starttime': start.strftime(api_date_format),
    'format': 'json',
    'id': 'BOU',
    'sampling_period': 1,
    'type': 'adjusted'
}
api_url = f'https://geomag.usgs.gov/ws/data/'
res = requests.get(api_url, params)

usgs_path = f'../logs/usgs/usgs_{start_file}__{end_file}.json'
with open(usgs_path, 'w') as fp:
    if res.status_code == 200:
        print('Successfully got USGS data, writing to file...')
        fp.write(res.text)
    else:
        print('Failed to get USGS data...')
        fp.write('{"error": ' + res.status_code + '}')

plot_log(log_file)
plot_log_fft(log_file, max_freq=1)
