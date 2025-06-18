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

data_dir = '/home/welling/RadioNeutrino/data/pueo/flavor/noise/'
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
filter_band = [.2, 1.]
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
n_events = 2000
n_channels = 96
n_templates = templateHelper.get_n_templates()
n_samples = int(1024 * upsampling_factor)
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
    beamformed_max_correlations[0, i_template, i_event] = np.max(np.abs(scipy.signal.hilbert(correlationHelper.correlate(
      beam_wf,
      i_template
    ))))
    beamformed_max_correlations[1, i_template, i_event] = np.max(np.abs(scipy.signal.hilbert(correlationHelper.correlate(
      beam_wf2,
      i_template
    ))))

print('Noise RMS: ', np.mean(noise_rms))
fig2 = plt.figure(figsize=(8, 12))
bins2 = np.linspace(np.min(beamformed_max_correlations), np.max(beamformed_max_correlations), 50)
max_corr_results = np.zeros((n_templates*2+1, bins2.shape[0]-1))
max_corr_results[0] = bins2[1:]
max_corr_thresholds = np.zeros(n_templates)
for i_template in range(n_templates):
  ax2_1 = fig2.add_subplot(n_templates, 2, 2*i_template + 1)
  ax2_2 = fig2.add_subplot(n_templates, 2, 2*i_template + 2)
  ax2_1.hist(
    beamformed_max_correlations[0, i_template],
    bins=bins2
  )
  ax2_1.hist(
    beamformed_max_correlations[1, i_template],
    bins=bins2
  )
  hist2 = ax2_2.hist(
    beamformed_max_correlations[0, i_template],
    bins=bins2,
    density=True,
    cumulative=-1,
    histtype='step'
  )
  hist3 = ax2_2.hist(
    beamformed_max_correlations[1, i_template],
    bins=bins2,
    density=True,
    cumulative=-1,
    histtype='step'
  )
  max_corr_results[i_template+1] = hist2[0]
  max_corr_results[i_template+1 + n_templates] = hist3[0]
  max_corr_thresholds[i_template] = np.percentile(beamformed_max_correlations[0, i_template], 99)
  ax2_1.grid()
  ax2_2.grid()
  ax2_2.set_yscale('log')
fig2.tight_layout()
fig2.savefig('plots/max_corr_background.png')
np.savetxt(
  'max_corr_beamformed.csv',
  max_corr_results,
  delimiter=', '
  )
np.savetxt(
  'beamformed_corr_thresholds.csv',
  max_corr_thresholds,
  delimiter=', '
)
