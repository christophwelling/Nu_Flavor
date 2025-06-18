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

parser = argparse.ArgumentParser()
parser.add_argument("run", type=int)
args = parser.parse_args()

data_dir = '/home/welling/RadioNeutrino/data/pueo/flavor/mu/'
data_file = data_dir+'run'+str(args.run)+'/IceFinal_'+str(args.run)+'_allTree.root'
antPosFile = '/home/welling/Software/pueo/usr/share/pueo/geometry/jun25/qrh.dat'

upsampling_factor = 10.
beamforming_degrees = 50.
filter_band = [.2, 1.]
# filter_band = None
noise_rms = .0219
amp_data = np.genfromtxt(
  '/home/welling/Software/pueo/usr/share/pueo/responses/signalChainMI/PUEO_SignalChainMI_0.csv',
  delimiter=','
)
amp_response = amp_data[:, 1] * np.exp(1.j*amp_data[:, 2])
amp_response = np.fft.irfft(amp_response)

dataReader = helpers.data_reader.DataReader(
  data_file,
  antPosFile,
  upsampling_factor,
  filter_band
)
templateHelper = helpers.template_helper.templateHelper(
  '../templates/templates.csv',
  'beamformed_corr_thresholds.csv',
  upsampling_factor,
  filter_band,
  True
)
correlationHelper = helpers.correlation_helper.correlationHelper(
  templateHelper
)
polHelper = helpers.polarization_estimator.polarizationEstimator(
  templateHelper
)
corr_quant_data = np.genfromtxt(
  'max_corr_beamformed.csv',
  delimiter=','
)
pulseCounter = helpers.pulse_counter.pulseCounter(
  int(10 * upsampling_factor)
)
probability_thresholds = np.array([.05, 1.e-2, 1.e-3])
found_weights = np.zeros((dataReader.get_n_events(), probability_thresholds.shape[0] + 1))
for i_event in range(dataReader.get_n_events()):
  plt.close('all')
  fig1 = plt.figure(figsize=(18, 12))
  ax1_0 = fig1.add_subplot(411)
  ax1_1 = fig1.add_subplot(412)
  ax1_2 = fig1.add_subplot(413)
  ax1_3 = fig1.add_subplot(414)

  dataReader.read_event(i_event)
  rf_dir = dataReader.get_signal_direction()
  max_channel = dataReader.get_max_channel()
  times = dataReader.get_times()
  weight = dataReader.get_event_weight()
  beam_wf = np.zeros((2, dataReader.get_n_samples()))
  noiseless_wf = np.zeros((2, dataReader.get_n_samples()))
  noiseless_wf[0] = dataReader.get_waveform(
    max_channel[0],
    0,
    True
  )
  noiseless_wf[1] = dataReader.get_waveform(
    max_channel[0],
    1,
    True
  )
  noisy_wf = dataReader.get_waveform(
    *max_channel,
    False
  )
  template, threshold, i_template = templateHelper.pick_template(beam_wf[max_channel[1]])
  dataReader.dedisperse(amp_response)  
  beam_wf[0] = dataReader.beamform(
    beamforming_degrees * np.pi / 180.,
    rf_dir,
    0,
    max_channel[0]
  )
  beam_wf[1] = dataReader.beamform(
    beamforming_degrees * np.pi / 180.,
    rf_dir,
    1,
    max_channel[0]
  )
  polarization = polHelper.estimate_polarization(beam_wf, max_channel[1])[0]
  pol_sign = np.sin(polarization) * np.cos(polarization)
  pol_sign /= np.abs(pol_sign)

  dedispersed_noiseless = dataReader.get_waveform(
    *max_channel,
    True
  )
  corr = correlationHelper.correlate(
    beam_wf[max_channel[1]],
    i_template
  )
  corr2 = correlationHelper.correlate(
    .5 * (beam_wf[0] + pol_sign * beam_wf[1]),
    i_template
  )
  corr_envelope = np.abs(scipy.signal.hilbert(corr))
  corr_envelope2 = np.abs(scipy.signal.hilbert(corr2))
  probs = np.zeros((2, corr.shape[0]))
  for ii in range(probs.shape[1]):
    probs[0, ii] = corr_quant_data[i_template+1, np.argmin(np.abs(corr_envelope[ii]-corr_quant_data[0]))]
    probs[1, ii] = corr_quant_data[i_template + templateHelper.get_n_templates()+1, np.argmin(np.abs(corr_envelope2[ii]-corr_quant_data[0]))]
  probs[probs==0] = 1. / 2000.
  ax1_0.plot(
    times,
    (noiseless_wf[0]) / noise_rms,
    alpha=1.
  )
  ax1_0.plot(
    times,
    (noiseless_wf[1]) / noise_rms,
    alpha=.5
  )
  ax1_1.plot(
    times,
    beam_wf[max_channel[1]]
  )
  ax1_1.plot(
    times,
    dedispersed_noiseless / np.max(dedispersed_noiseless) * np.max(beam_wf),
    alpha=.5
  )
  t_max = times[np.argmax(noiseless_wf[max_channel[1]])]
  t_range = [t_max - 512/3., t_max+512/3.]
  ax1_0.axvspan(*t_range, color='k', alpha=.1)
  ax1_1.axvspan(*t_range, color='k', alpha=.1)
  ax1_2.axvspan(*t_range, color='k', alpha=.1)
  ax1_3.axvspan(*t_range, color='k', alpha=.1)
  ax1_0.grid()
  # ax1_0.set_yscale('log')
  ax1_0.set_ylim([-4, 4])
  ax1_1.grid()
  ax1_2.plot(
    times,
    np.abs(corr),
    color='C0'
  )
  ax1_2.plot(
    times,
    corr_envelope,
    color='C0',
    alpha=.5
  )
  ax1_2.plot(
    times,
    np.abs(corr2),
    color='C1',
    alpha=.5
  )
  ax1_2.plot(
    times,
    corr_envelope2,
    color='C1',
    alpha=.25
  )
  ax1_2.grid()
  ax1_3.plot(
    times,
    probs[0],
    linestyle='-',
    linewidth=4,
    marker='none'
  )
  if np.abs(np.cos(polarization)) < .2 or np.abs(np.sin(polarization)) < .2:
    i_peak = np.argmax(beam_wf[max_channel[1]])
    if i_peak < 512 * upsampling_factor:
      i_peak = int(512 * upsampling_factor)
    if i_peak + 512 * upsampling_factor >= corr_envelope.shape[0]:
      i_peak = int(corr_envelope.shape[0] - 512 * upsampling_factor - 1)
    prob2_col = 'k'
    window = [int(i_peak-512*upsampling_factor), int(i_peak+512*upsampling_factor)]
    pulses = pulseCounter.count_pulses(
      corr_envelope[window[0]:window[1]],
      probs[0, window[0]:window[1]],
      probability_thresholds
    )
    for pulse in pulses:
      pulse[:2] += window[0]
  else:
    i_peak = np.argmax(beam_wf[0] + pol_sign * beam_wf[1])
    if i_peak < 512 * upsampling_factor:
      i_peak = int(512 * upsampling_factor)
    if i_peak + 512 * upsampling_factor >= corr_envelope2.shape[0]:
      i_peak = int(corr_envelope2.shape[0] - 512 * upsampling_factor - 1)
    prob2_col = 'k'
    window = [int(i_peak-512*upsampling_factor), int(i_peak+512*upsampling_factor)]
    prob2_col = 'C1'
    pulses = pulseCounter.count_pulses(
      corr_envelope2[window[0]:window[1]],
      probs[1, window[0]:window[1]],
      probability_thresholds
    )
    for pulse in pulses:
      pulse[:2] += window[0]
  found_weights[i_event, 0] = weight
  for i_pulse, pulse in enumerate(pulses):
    if i_pulse > 0:
      found_weights[i_event, pulse[2]+1:] = weight
    ax1_1.axvspan(
      times[pulse[0]] + np.argmax(template) / 3. / upsampling_factor,
      times[pulse[1]] + np.argmax(template) / 3. / upsampling_factor,
      color='C{}'.format(pulse[2]),
      alpha=.2
    )
    ax1_3.axvspan(
      times[pulse[0]],
      times[pulse[1]],
      color='C{}'.format(pulse[2]),
      alpha=.2
    )
  ax1_3.plot(
    times,
    probs[1],
    alpha=1.,
    linestyle='-',
    marker='None',
    color=prob2_col
  )
  ax1_3.axhline(.05, color='r', linestyle=':', alpha=.5)
  ax1_3.axhline(.01, color='r', linestyle=':', alpha=.5)
  ax1_3.axhline(.001, color='r', linestyle=':', alpha=.5)
  ax1_3.grid()
  # ax1_3.set_xlim(t_range)
  ax1_3.set_ylim([1.e-4, 1.])
  ax1_3.set_yscale('log')
  fig1.tight_layout()
  fig1.savefig('plots/correlation_search/correlations_{}.png'.format(i_event))

  ax1_0.set_xlim(t_range)
  ax1_1.set_xlim(t_range)
  ax1_2.set_xlim(t_range)
  ax1_3.set_xlim(t_range)

  fig1.tight_layout()
  fig1.savefig('plots/correlation_search_zoomed/correlations_{}.png'.format(i_event))
sorted_thresholds = np.sort(probability_thresholds)
for i_threshold in range(probability_thresholds.shape[0]):
  found_fraction = np.sum(found_weights[:, i_threshold+1]) / np.sum(found_weights[:, 0])
  print('Pulses found at {} confidence level: '.format(sorted_thresholds[i_threshold]), found_fraction)