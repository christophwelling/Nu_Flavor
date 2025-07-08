import numpy as np
import matplotlib.pyplot as plt
import json
import glob

results_files = glob.glob('pulse_search_results*.json')

n_thresholds = 3

energy_bins = np.arange(18, 20.5, .5)
weight_sums = np.zeros(energy_bins.shape[0])
efficiencies = np.zeros((n_thresholds+1, energy_bins.shape[0]))
for i_file, filename in enumerate(results_files):
  data = json.load(open(filename))
  for i_event, event_data in enumerate(data['events']):
    i_energy_bin = np.argmin(np.abs(energy_bins - np.log10(event_data['nu_energy'])))
    weight_sums[i_energy_bin] += event_data['absorption_weight']
    if len(event_data['sub_events'])>= 2:
      efficiencies[0, i_energy_bin] += 1. # event_data['absorption_weight']
    pulses_found = np.zeros(n_thresholds)
    for i_sub_event, sub_event in enumerate(event_data['sub_events']):
      for i_pulse, pulse in enumerate(sub_event['pulses_found']):
        pulses_found[pulse['threshold']:] += 1
    for i_thresh in range(n_thresholds):
      if np.sum(pulses_found[i_thresh]) >= 2:
        efficiencies[i_thresh+1, i_energy_bin] +=  event_data['absorption_weight']

fig1 = plt.figure(figsize=(8, 6))
weight_filter = weight_sums > 0
ax1_1 = fig1.add_subplot(111)
for i in range(n_thresholds +1):
  ax1_1.scatter(
    energy_bins[weight_filter],
    efficiencies[i][weight_filter] / weight_sums[weight_filter],
    label=i
  )
ax1_1.legend()
ax1_1.grid()
fig1.tight_layout()
fig1.savefig('plots/efficiencies.png')
    