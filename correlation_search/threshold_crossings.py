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

upsampling_factor = 10
filter_band = [.3, .7]
dataReader = helpers.data_reader.DataReader(
  data_file,
  antPosFile,
  upsampling_factor,
  filter_band
)
amp_data = np.genfromtxt(
  '/home/welling/Software/pueo/usr/share/pueo/responses/signalChainMI/PUEO_SignalChainMI_0.csv',
  delimiter=','
)
amp_response = amp_data[:, 1] * np.exp(1.j*amp_data[:, 2])
amp_response = np.fft.irfft(amp_response)

templateHelper = helpers.template_helper.templateHelper(
  '../templates/templates.csv',
  'corr_stdv.csv',
  upsampling_factor,
  filter_band,
  True
)
correlationHelper = helpers.correlation_helper.correlationHelper(
  templateHelper
)
corr_stds = np.genfromtxt(
  'corr_stdv.csv',
  delimiter=','
)
thresholds = np.array([2, 3, 4])
for i_event in range(dataReader.get_n_events()):
  plt.close('all')
  fig1 = plt.figure(figsize=(12, 12))
  ax1_0 = fig1.add_subplot(211)

  dataReader.read_event(i_event)
  dataReader.dedisperse(amp_response)
  rf_dir = dataReader.get_signal_direction()
  max_channel = dataReader.get_max_channel()
  times = dataReader.get_times()
  noiseless_wf = dataReader.get_waveform(
    *max_channel,
    True
  )
  noisy_wf = dataReader.get_waveform(
    *max_channel,
    False
  )
  template, thr, i_template = templateHelper.pick_template(
    dataReader.get_waveform(*max_channel)
  )
  waveforms = dataReader.get_waveforms(
      40. * np.pi / 180.,
      rf_dir,
      0
    )
  correlations = np.zeros((2, waveforms.shape[0], waveforms.shape[1]))
  ones = np.ones(correlations.shape[1])
  th_crossings = np.zeros((2, thresholds.shape[0], correlations.shape[2]))
  for i_pol in range(2):
    waveforms = dataReader.get_waveforms(
      40. * np.pi / 180.,
      rf_dir,
      i_pol
    )
    for i_wf in range(waveforms.shape[0]):
      correlations[i_pol, i_wf] = correlationHelper.correlate(
        waveforms[i_wf],
        i_template
      )
    for i_threshold, threshold in enumerate(thresholds):
      for ii in range(correlations.shape[2]):
        th_crossings[i_pol, i_threshold, ii] = np.sum(ones[np.abs(correlations[i_pol, :, ii])>threshold*corr_stds[i_template]])
  ax1_0.plot(
    times,
    noisy_wf
  )
  ax1_0.plot(
    times,
    noiseless_wf,
    alpha=.5
  )
  t_max = times[np.argmax(noiseless_wf)]
  window_size = 1024 / 3.
  t_range = np.array([t_max - 512/3., t_max+512/3.])
  ax1_0.set_xlim([t_max-.6*window_size, t_max+.6*window_size])
  ax1_0.axvspan(t_max-.5*window_size, t_max+.5*window_size, color='k', alpha=.1)
  for i_threshold in range(thresholds.shape[0]):
    ax1_1 = fig1.add_subplot(2*thresholds.shape[0], 1, i_threshold+thresholds.shape[0]+1)
    # ax1_1.plot(
    #   times,
    #   th_crossings[max_channel[1], i_threshold],
    #   marker='o',
    #   linestyle='none',
    #   alpha=.2
    # )
    ax1_1.plot(
      times,
      np.sum(th_crossings[:, i_threshold], axis=0),
      marker='o',
      linestyle='none',
      alpha=.2
    )
    ax1_1.set_xlim([t_max-.6*window_size, t_max+.6*window_size])
    ax1_1.axvspan(t_max-.5*window_size, t_max+.5*window_size, color='k', alpha=.1)
    ax1_1.grid()
  ax1_0.grid()
  fig1.tight_layout()
  fig1.savefig('plots/threshold_crossings/crossings_{}.png'.format(i_event))
