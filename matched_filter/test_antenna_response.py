import numpy as np
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import helpers.antenna_helper

ant_response = helpers.antenna_helper.AntennaHelper()

az = 40
el = 0
degrad = np.pi / 180.
boresight = np.array([-1, 0, 0])
signal_direction = np.array([
  np.cos(az*degrad) * np.cos(el*degrad),
  np.sin(az*degrad) * np.cos(el*degrad),
  np.sin(el*degrad)
  ])
response = ant_response.get_antenna_response_for_direction(
  signal_direction,
  boresight,
  0
)
freqs = np.fft.rfftfreq(1024, 1./3.)
fig1 = plt.figure(figsize=(8, 6))
ax1_1 = fig1.add_subplot(111)
ax1_1.plot(
  freqs,
  10. * np.log10(np.abs(response)**2 * 4. * np.pi * (freqs*1.e9)**2 / (3.e8)**2)
)
ax1_1.set_ylim([0, None])
ax1_1.grid()
fig1.tight_layout()
fig1.savefig('antenna_response.png')

