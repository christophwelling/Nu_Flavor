import numpy as np
import matplotlib.pyplot as plt

import argparse
import scipy.signal
import sys
sys.path.append('../')
import helpers.data_reader 
import helpers.template_helper
import helpers.correlation_helper

parser = argparse.ArgumentParser()
parser.add_argument("run", type=int)
args = parser.parse_args()

data_dir = '/home/welling/RadioNeutrino/data/pueo/flavor/'
data_file = data_dir+'run'+str(args.run)+'/IceFinal_'+str(args.run)+'_allTree.root'
antPosFile = '/home/welling/Software/pueo/usr/share/pueo/geometry/photogrammetry/pueoPhotogrammetry.csv'
amp_data = np.genfromtxt(
  '/home/welling/Software/pueo/usr/share/pueo/responses/signalChainMI/PUEO_SignalChainMI_0.csv',
  delimiter=','
)
amp_response = amp_data[:, 1] * np.exp(1.j*amp_data[:, 2])
amp_response = np.fft.irfft(amp_response)
upsampling_factor = 10
beamforming_degrees = 40.
filter_band = [.3, .7]
# filter_band = None
dataReader = helpers.data_reader.DataReader(
  data_file,
  antPosFile,
  upsampling_factor,
  filter_band
)
templateHelper = helpers.template_helper.templateHelper(
  '../templates/templates.csv',
  None,
  upsampling_factor,
  filter_band,
  True
)
correlationHelper = helpers.correlation_helper.correlationHelper(
  templateHelper
)
n_events = 500
n_channels = 96
n_templates = templateHelper.get_n_templates()
n_samples = int(1024 * upsampling_factor)
correlations = np.zeros((2, n_events, n_channels, n_templates, n_samples//4))
beamformed_max_correlations = np.zeros((2, n_templates, n_events))
noise_rms = np.zeros((2, n_events, n_channels))
rf_dir = np.array([1, 0, 0])
for i_event in range(n_events):
  plt.close('all')
  dataReader.read_event(i_event)
  dataReader.dedisperse(amp_response)
  max_channel = dataReader.get_max_channel()
  times = dataReader.get_times()
  for i_pol in range(2):
    for i_channel in range(n_channels):
      waveform = dataReader.get_waveform(
        i_channel,
        i_pol,
        False
      )
      noise_rms[i_pol, i_event, i_channel] = np.sqrt(np.mean(waveform**2))
      for i_template in range(n_templates):
        correlations[i_pol, i_event, i_channel, i_template] = correlationHelper.correlate(
          waveform,
          i_template
        )[n_samples//4:n_samples//2]
    beam_wf = dataReader.beamform(
      beamforming_degrees * np.pi / 180.,
      rf_dir,
      0,
      0
    )
    beam_wf2 = .5 * (beam_wf + dataReader.beamform(
      beamforming_degrees * np.pi / 180.,
      rf_dir,
      1,
      0
    ))
    for i_template in range(n_templates):
      beamformed_max_correlations[0, i_template, i_event] = np.max(np.abs(correlationHelper.correlate(
        beam_wf,
        i_template
      )))
      beamformed_max_correlations[1, i_template, i_event] = np.max(np.abs(correlationHelper.correlate(
        beam_wf2,
        i_template
      )))

print('Noise RMS: ', np.mean(noise_rms))
fig1 = plt.figure(figsize=(12, 10))
max_corr = np.max(np.abs(correlations))
bins = np.arange(-max_corr, max_corr, max_corr / 200.)
template_size = templateHelper.get_template_size()
corr_std = np.zeros(n_templates)
for i_template in range(n_templates):
  ax1_1 = fig1.add_subplot(n_templates, 1, i_template+1)
  hist = ax1_1.hist(
    correlations[:, :, :, i_template, :-template_size].flatten(),
    bins=bins,
    density=True
  )
  ax1_1.grid()
  ax1_1.set_yscale('log')
  ax1_1.set_ylim([np.min(hist[0][hist[0]>0]), None])
  corr_std[i_template] = np.sqrt(np.mean(correlations[:, :, :, i_template, :-template_size]**2))
  ax1_1.plot(
    bins,
    1./np.sqrt(2.*np.pi)/corr_std[i_template] * np.exp(-.5*bins**2/corr_std[i_template]**2),
    color='k'
  )
fig1.tight_layout()
fig1.savefig('plots/background_correlations.png')
np.savetxt(
  'corr_stdv.csv',
  corr_std,
  delimiter=','
)
