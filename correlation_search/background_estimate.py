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
parser.add_argument("run", type=int)
args = parser.parse_args()

upsampling_factor = 10.
n_events = 200
n_samples = int(1024*upsampling_factor)
beamforming_degrees = 40.
antPosFile = '/home/welling/Software/pueo/usr/share/pueo/geometry/jun25/qrh.dat'
filter_band = [.2, 1.]
# filter_band = None
folders = []
rf_dir = np.array([1, 0, 0])
data_dir = '/home/welling/RadioNeutrino/data/pueo/flavor/noise/'
data_file = data_dir+'run'+str(args.run)+'/IceFinal_'+str(args.run)+'_allTree.root'
templateHelper = helpers.template_helper.templateHelper(
  '../templates/templates.csv',
  'beamformed_corr_thresholds.csv',
  upsampling_factor,
  filter_band,
  True
)
dataReader = helpers.data_reader.DataReader(
  data_file,
  antPosFile,
  upsampling_factor,
  filter_band
)


pulseFinder = helpers.pulse_finder.pulseFinder(
  antPosFile,
  upsampling_factor,
  filter_band,
  templateHelper,
  'max_corr_beamformed.csv'
)
max_correlations = np.zeros((2, templateHelper.get_n_templates(), min(n_events, dataReader.get_n_events())))
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
      )[:template.shape[0]-1]))
      max_correlations[i_pol, i_template, i_event] = np.max(correlations[i_pol])

fig1 = plt.figure(figsize=(12, 8))