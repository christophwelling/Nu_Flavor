import numpy as np
import matplotlib.pyplot as plt

import argparse
import scipy.signal
import sys
sys.path.append('../')
import helpers.data_reader 
import helpers.template_helper
import helpers.correlation_helper
import helpers.polarization_estimator
import helpers.pulse_counter
import helpers.pulse_finder
import glob
import os.path

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("run", type=int)
args = parser.parse_args()

upsampling_factor = 10.
n_events = 3000
n_samples = int(1024*upsampling_factor)
beamforming_degrees = 70.
antPosFile = '/home/welling/Software/pueo/usr/share/pueo/geometry/jun25/qrh.dat'
filter_band = [.2, 1.]
# filter_band = None
folders = []
rf_dir = np.array([1, 0, 0])
data_file = args.path +'/run'+str(args.run)+'/IceFinal_'+str(args.run)+'_allTree.root'
templateHelper = helpers.template_helper.templateHelper(
  '../templates/templates.csv',
  None,
  upsampling_factor,
  filter_band,
  True
)
dataReader = helpers.data_reader.DataReader(
  data_file,
  antPosFile,
  1,
  filter_band
)


pulseFinder = helpers.pulse_finder.pulseFinder(
  antPosFile,
  upsampling_factor,
  filter_band,
  templateHelper,
  None
)
max_correlations = np.zeros((2, templateHelper.get_n_templates(), min(n_events, dataReader.get_n_events())))
noise_rms = np.zeros(min(n_events, dataReader.get_n_events()))
for i_event in range(min(n_events, dataReader.get_n_events())):
  dataReader.read_event(i_event)
  channel_ids = dataReader.get_close_channels(
      beamforming_degrees * np.pi / 180.,
      0
    )
  wf = np.zeros((2, channel_ids.shape[0], dataReader.get_n_samples()))
  for i_pol in range(2):
    for i_ch in range(channel_ids.shape[0]):
      wf[i_pol, i_ch] = dataReader.get_waveform(
        channel_ids[i_ch],
        i_pol,
        False
      )
  noise_rms[i_event] = np.sqrt(np.mean(wf[0, 0]**2))
  pulseFinder.set_waveforms(
    wf,
    channel_ids
  )
  pulseFinder.dedisperse()
  beams = pulseFinder.beamform(
    rf_dir,
    0,
    True
  )
  correlations = np.zeros((2, beams.shape[1]))
  for i_template in range(templateHelper.get_n_templates()):
    template = templateHelper.get_template(i_template)
    for i_pol in range(2):
      correlations[i_pol] = np.abs(scipy.signal.hilbert(np.correlate(
        beams[i_pol],
        template,
        'full'
      )[template.shape[0]-1:]))
      max_correlations[i_pol, i_template, i_event] = np.max(correlations[i_pol])

n_bins = 50
bins = np.linspace(np.min(max_correlations), np.max(max_correlations),  n_bins)
n_templates = templateHelper.get_n_templates()
background_output = np.zeros((n_templates+1, n_bins-1))
fig1 = plt.figure(figsize=(12, 8))
for i_template in range(n_templates):
  ax1_1 = fig1.add_subplot(n_templates, 2, i_template*2+1)
  ax1_2 = fig1.add_subplot(n_templates, 2, i_template*2+2)
  ax1_1.hist(
    max_correlations[:, i_template, :].flatten(),
    bins=bins
  )
  entries, bin_edges, patches = ax1_2.hist(
    max_correlations[:, i_template, :].flatten(),
    density=True,
    cumulative=-1,
    bins=bins
  )
  background_output[0] = bin_edges[:-1] + .5 * (bin_edges[1] - bin_edges[0])
  background_output[i_template+1] = entries
  ax1_2.set_yscale('log')
  ax1_1.grid()
  ax1_2.grid()
fig1.tight_layout()
fig1.savefig('plots/background.png')
np.savetxt(
  'background_correlations.csv',
  background_output,
  delimiter=', '
)
np.savetxt(
  'noise_rms.csv',
  noise_rms,
  delimiter=', '
)