import numpy as np

RESISTANCE = 200  # OHMS
PCB_INVERSION = np.array([-1.02923, -1.02923, -1.02923, -1.032])  # No units
MAGNETOMETER_SCALING_FACTOR = 3.6e-5  # Tesla / Volt

import os
from fieldline_coil_system import compute_magnitudes, merge_log_usgs
mag_files = list(map(lambda x: x.replace('mag_', '').replace('.csv', ''),
                     filter(lambda x: x.startswith('mag_'), os.listdir('../logs/readings/lab'))))
usgs_files = list(map(lambda x: x.replace('usgs_', '').replace('.json', ''),
                      filter(lambda x: x.startswith('usgs_'), os.listdir('../logs/usgs/'))))

overlapping = set(mag_files) & set(usgs_files)

def sf(name):
    df = merge_log_usgs(name)
    df[['USGS - X', 'USGS - Y', 'USGS - Z']] *= 1e-9  # in nT, convert to tesla
    df['rmag'] = compute_magnitudes(df['x', 'y', 'z'])  # in volts
    df['umag'] = compute_magnitudes(df['USGS - X', 'USGS - Y', 'USGS - Z'])
    return np.mean(df['umag'] / df['rmag'])  # get T/V scaling factor from usgs data

scaling_factors = []
for ov in overlapping:
    scaling_factors.append(sf(ov))  # do it for lots of runs. some data is bad, but it's obvious
    # distinct groups of ~3.6e-5, ~1.7e-5, and ~0.2.

