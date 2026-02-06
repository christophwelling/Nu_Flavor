import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('flavor', type=str)
args = parser.parse_args()
results_folders = glob.glob('/home/welling/RadioNeutrino/data/pueo/flavor/found_pulses_old/{}/*'.format(args.flavor))

fontsize=12
thresholds = np.array([.01, 1.e-3])
n_thresholds = thresholds.shape[0]

energy_bins = np.arange(18, 20.5, .25)
weight_sums = np.zeros(energy_bins.shape[0])
efficiencies = np.zeros((2, n_thresholds, energy_bins.shape[0]))
multi_triggers = np.zeros(energy_bins.shape[0])
efficiency_any = np.zeros((n_thresholds, energy_bins.shape[0]))
for folder in results_folders:
  results_files = glob.glob(folder+'/*.json')
  for filename in results_files:
    event_data = json.load(open(filename))
    i_energy_bin = np.argmin(np.abs(energy_bins - np.log10(event_data['nu_energy'])))
    weight_sums[i_energy_bin] += event_data['weight']
    n_sub_events = 0
    for sub_event in event_data['sub_events']:
      n_sub_events += 1
    if n_sub_events >= 2:
      multi_triggers[i_energy_bin] += event_data['weight']
    for i_threshold, threshold in enumerate(thresholds):
      multi_pulse_found = False
      for sub_event in event_data['sub_events']:
        n_pulses = 0
        for pulse in sub_event['pulses_found']:
          if pulse['min_probability'] <= threshold:
            n_pulses += 1
        if n_pulses >= 2:
          multi_pulse_found = True
      if multi_pulse_found:
        efficiencies[0, i_threshold, i_energy_bin] += event_data['weight']
      if multi_pulse_found or n_sub_events >= 2:
        efficiencies[1, i_threshold, i_energy_bin] += event_data['weight']
fig1 = plt.figure(figsize=(8, 6))
weight_filter = weight_sums > 0
ax1_1 = fig1.add_subplot(111)
for i in range(n_thresholds):
  ax1_1.plot(
    energy_bins[weight_filter]+.00*i,
    efficiencies[0, i][weight_filter] / weight_sums[weight_filter],
    label='{:.1f}%'.format(thresholds[i]*100),
    alpha=1.,
    color='C{}'.format(i),
    marker='s'
  )
#   ax1_1.plot(
#     energy_bins[weight_filter]+.00*i,
#     efficiencies[1, i][weight_filter] / weight_sums[weight_filter],
#     label='{:.1f}% + multi-trigger'.format(thresholds[i]*100),
#     alpha=1.,
#     color='C{}'.format(i),
#     marker='x',
#     linestyle='--'
#   )
# ax1_1.plot(
#   energy_bins[weight_filter]-.00,
#   multi_triggers[weight_filter] / weight_sums[weight_filter],
#   color='k',
#   linestyle=':',
#   label='multi-trigger',
#   marker='o'
# )

ax1_1.set_ylim([-.01, .65])
ax1_1.legend(ncols=3, fontsize=fontsize)
ax1_1.grid()
ax1_1.set_xlabel(r'$log_{10}(E_\nu/eV)$', fontsize=fontsize)
ax1_1.set_ylabel('efficiency', fontsize=fontsize)
fig1.tight_layout()
fig1.savefig('plots/efficiencies_{}.png'.format(args.flavor))

fig2 = plt.figure(figsize=(8, 6))
ax2_1 = fig2.add_subplot(111)
ax2_1.plot(
  energy_bins[weight_filter],
  weight_sums[weight_filter],
  marker='o',
  linestyle='--'
)
ax2_1.set_ylim([0, 15.])
ax2_1.grid()
fig2.tight_layout()
fig2.savefig('plots/weight_sums_{}.png'.format(args.flavor))
