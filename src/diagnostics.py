import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analysis import *


df = read_log('./dataset/mag_2022-02-24T12-15-30__2022-02-24T12-20-30.csv')
sampling_rate = 120
df = resample_dataframe(df, sampling_rate, 'mean', True)
baseline = df.loc[[0, 1, 2, 3, 4]] # first 5 seconds of data, will act as baseline readings

# setup an example simulation to get the magnetic field matrix used to cancel out the field
# Using this matrix, we have the unit vector in units of T/A for each coil. Use this vector as the baseline and
# scale it using a sin wave to mimic applying a current to the coil. Overlay this sin wave (plus noise?) over
# the coil dataframe.
fn_sin = lambda l, f, a, N: a * np.sin(np.linspace(0, l*2*np.pi, num=N) * f) # helper function for sin waves

# define environment
wall_1 = {
  'center': [0, 0, 18.5 * 2.54 / 2.0],
  'shape': [1, 2],
  'a1': 16.0*2.54/2.0,
  'b1': 16.0*2.54/2.0,
  'coil_spacing': 1.0 * 2.54,
  'rotation_axis': None, 'theta': 0
}
wall_2 = {**wall_1, 'center': [0, 0, -18.5 * 2.54 / 2.0]}

# compute the contribution of all the coils on the magnetomter
B = get_full_b(wall_1, wall_2, [0, 0, 0]).T # transpose to get rows as each coil's unit vector
v = B[0]  # use one coil's contribution

# generate a noisy sin wave, amplitude goes from +/- 50 milli-amps w/ 1 milli-amp std of noise
s = fn_sin(5, 10, 50e-3, sampling_rate*5) + np.random.normal(loc=0, scale=1e-3, size=sampling_rate*5)

# use the next 5 seconds to overlay the coil data over
coil = df.loc[[5, 6, 7, 8, 9]]
# 'turn on' the coil
coil['x'] += v[0] * s
coil['y'] += v[1] * s
coil['z'] += v[2] * s

# compare the baseline to when the coil is on
freq = 10
freq_sweep = 1.5
bfreq, bdense, cfreq, cdense = coil_impact(baseline, coil, freq, freq_sweep, sampling_rate)
plt.plot(bfreq, bdense, label='Ambient Measurements')
plt.plot(cfreq, cdense, label='Coil Measurements')
plt.ylabel(r'$T^2$')
plt.xlabel('Hz')
plt.legend()
plt.show()


def coil_impact(baseline, coil, freq, freq_sweep, sampling_rate):
    baseline_mag = compute_magnitudes(baseline, ['x', 'y', 'z'])
    coil_mag = compute_magnitudes(coil, ['x', 'y', 'z'])

    baseline_freqs, baseline_dens = fft_signal(baseline_mag, sampling_rate)
    baseline_idx = np.where((freq - freq_sweep <= baseline_freqs) &
                            (baseline_freqs <= freq + freq_sweep))[0]
    baseline_freqs = baseline_freqs[baseline_idx]
    baseline_dens = baseline_dens[baseline_idx]

    coil_freqs, coil_dens = fft_signal(coil_mag, sampling_rate)
    coil_idx = np.where((freq - freq_sweep <= coil_freqs) &
                        (coil_freqs <= freq + freq_sweep))[0]
    coil_freqs = coil_freqs[coil_idx]
    coil_dens = coil_dens[coil_idx]

    return baseline_freqs, baseline_dens, coil_freqs, coil_dens





