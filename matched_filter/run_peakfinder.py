import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import argparse
import sys
import os
import glob
sys.path.append('../')
import pickle
import helpers.peakfinder

parser = argparse.ArgumentParser()
parser.add_argument('flavor', type=str)
parser.add_argument('run', type=int)
args = parser.parse_args()

filenames = glob.glob('results/{}/run{}/*.pkl'.format(args.flavor, args.run))
peakFinder = helpers.peakfinder.PeakFinder()

for i_file, filepath in enumerate(filenames):
  plt.close('all')
  file = open(filepath, 'rb')
  pkl_data = pickle.load(file)
  filename = filepath.split('/')[-1][:-4]
  i_event = int(filename.split('_')[1])
  i_trigger = int(filename.split('_')[2])
  data = pkl_data['data']
  peaks = peakFinder.find_peaks(
    np.abs(scipy.signal.hilbert(data[4])),
    .2
  )
  shower_times = pkl_data['shower_times']
  shower_energies = pkl_data['shower_energies']
  peaks = np.array(peaks)
  peaks = peaks[peaks[:, 0].argsort()]
  fig1, ax1 = plt.subplots(5, 1, figsize=(18, 12), sharex=True)
  times = np.arange(data.shape[1]) / 3.
  for i_pol in range(2):
    ax1[i_pol].plot(
      times,
      data[i_pol]
    )
    ax1[i_pol].plot(
      times,
      data[i_pol+2]
    )
    ax1[i_pol].grid()
    ax1[i_pol].set_ylim([-3, 3])
    ax1[i_pol].set_xlim([times[0], times[-1]])
    ax1[i_pol].set_ylabel('U / RMS$_{noise}$')
    ax1[i_pol].axhline(-.5, color='k', alpha=.2, linestyle='--')
    ax1[i_pol].axhline(.5, color='k', alpha=.2, linestyle='--')
  ax1[2].plot(
    times,
    data[4]
  )
  ax1[2].plot(
    times,
    np.abs(scipy.signal.hilbert(data[4])),
    color='r',
    alpha=.3
  )
  for i_peak, peak in enumerate(peaks):
    ax1[2].axvspan(
      times[peak[0]],
      times[peak[1]],
      color='C{}'.format(i_peak % 3),
      alpha=.2
    )
    if np.min(data[5, peak[0]:peak[1]+1]) < 1.e-2:
     ax1[3].axvspan(
      times[peak[0]],
      times[peak[1]],
      color='C{}'.format(i_peak % 3),
      alpha=.2
    )
  ax1[2].set_ylabel('template correlation')
  ax1[3].plot(
    times,
    data[5]
  )
  ax1[2].grid()
  ax1[2].set_xlim([times[0], times[-1]])
  ax1[3].grid()
  ax1[3].set_yscale('log')
  ax1[3].set_ylim([5.e-4, 2])
  ax1[3].set_xlim([times[0], times[-1]])
  ax1[3].set_ylabel('probability')
  ax1[3].set_xlabel('t [ns]')
  for i_shower, shower_time in enumerate(shower_times):
    if times[0] < shower_time < times[-1]:
      ax1[3].axvline(
        shower_time,
        color='r',
        alpha=.3
      )
  ax1[4].scatter(
    shower_times,
    shower_energies
  )
  ax1[4].set_yscale('log')
  ax1[4].grid()
  ax1[4].set_xlim([times[0], times[-1]])
  ax1[4].set_xlabel('t [ns]')
  ax1[4].set_ylabel('$E_{shower}$ [eV]')
  ax1[4].set_ylim([1.e16-5, 1.e20])
  fig1.tight_layout()
  if not os.path.isdir('plots/peakfinder/{}'.format(args.flavor)):
    os.mkdir('plots/peakfinder/{}'.format(args.flavor))
  if not os.path.isdir('plots/peakfinder/{}/run{}'.format(args.flavor, args.run)):
    os.mkdir('plots/peakfinder/{}/run{}'.format(args.flavor, args.run))
  fig1.savefig('plots/peakfinder/{}/run{}/peakfinder_{}_{}.png'.format(
    args.flavor, 
    args.run,
    i_event,
    i_trigger
  ))

