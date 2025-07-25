import numpy as np
import sys
import gc
import os
import pickle
sys.path.append('../')
import helpers.data_reader
import helpers.peakfinder
import matched_filter_helper
import argparse
import plotting
import scipy.signal


parser = argparse.ArgumentParser()
parser.add_argument('flavor', type=str)
parser.add_argument('run', type=int)
args = parser.parse_args()

antenna_angle_cut = 50. * np.pi / 180.

filename = '~/RadioNeutrino/data/pueo/flavor/{}/run{}/IceFinal_{}_allTree.root'.format(args.flavor, args.run, args.run)
upsampling_factor = 1
n_samples = int(1024 * upsampling_factor)

dataReader = helpers.data_reader.DataReader(
  filename,
  None,
  upsampling_factor
)
mf_helper = matched_filter_helper.MatchedFilterHelper(
  '~/RadioNeutrino/data/pueo/flavor/noise/run99/IceFinal_99_allTree.root',
  upsampling_factor

)
mf_helper.calculate_noise_spectral_density()
peakFinder = helpers.peakfinder.PeakFinder()

for i_event in range(dataReader.get_n_events()):
  print(i_event, ' / ', dataReader.get_n_events())
  dataReader.read_event(i_event)
  signal_direction = dataReader.get_signal_direction()
  antennas = mf_helper.get_antenna_indices(signal_direction, antenna_angle_cut)
  times_ = dataReader.get_times()
  times = np.arange(n_samples) / 3. / upsampling_factor
  wf_ = np.zeros((2, len(antennas), times_.shape[0]))
  wf_noiseless_ = np.zeros_like(wf_)
  polarization_angle = dataReader.get_polarization_angle()
  viewing_angle = dataReader.get_viewing_angle()
  shower_energies = dataReader.get_energies()
  shower_had_fracs = dataReader.get_had_fracs()
  for i_ant, ant in enumerate(antennas):
    wf_[0, i_ant] = dataReader.get_waveform(ant, 0)
    wf_[1, i_ant] = dataReader.get_waveform(ant, 1)
    wf_noiseless_[0, i_ant] = dataReader.get_waveform(ant, 0, True)
    wf_noiseless_[1, i_ant] = dataReader.get_waveform(ant, 1, True)
  for i_trigger, trigger_time in enumerate(dataReader.get_trigger_times()):
    if trigger_time <= 0:
      continue
    trigger_index = np.argmin(np.abs(trigger_time - times_))
    if trigger_index < n_samples//2:
      trigger_index = n_samples//2
    if trigger_index > wf_.shape[2] - n_samples//2:
      trigger_index = wf_.shape[2] - n_samples//2
    wf = wf_[:, :, trigger_index-n_samples//2:trigger_index+n_samples//2]
    wf_noiseless = wf_noiseless_[:, :, trigger_index-n_samples//2: trigger_index+n_samples//2]
    shower_signal_times = dataReader.get_det_times() * 1.e9 - times_[trigger_index - n_samples//2] + 215
    max_channel = np.argmax(np.max(wf_noiseless, axis=(0, 2)))
    # plotting.plot_waveforms(
    #   i_event,
    #   i_trigger,
    #   times,
    #   wf,
    #   wf_noiseless,
    #   args.flavor,
    #   args.run
    # )
    template = mf_helper.generate_template(
      signal_direction,
      antennas,
      polarization_angle,
      viewing_angle
    )
    corr = mf_helper.apply_matched_filter(
      template,
      wf
    )
    probs, noise_rms = mf_helper.estimate_background_rate(
      template,
      antennas,
      np.sum(np.sum(corr, axis=0), axis=0),
      1000
    )
    if not os.path.isdir('results/{}'.format(args.flavor)):
      os.mkdir('results/{}'.format(args.flavor))
    if not os.path.isdir('results/{}/run{}'.format(args.flavor, args.run)):
      os.mkdir('results/{}/run{}'.format(args.flavor, args.run))
    results = np.zeros((6, times.shape[0]))
    results[:2] = wf[:, max_channel] / noise_rms
    results[2:4] = wf_noiseless[:, max_channel] / noise_rms
    results[4] = np.sum(corr, axis=(0, 1))
    results[5] = probs
    output = {
      'data': results,
      'shower_times': shower_signal_times,
      'shower_energies': shower_energies,
      'shower_hadfract': shower_had_fracs
    }
    outfile = open('results/{}/run{}/result_{}_{}.pkl'.format(args.flavor, args.run, i_event, i_trigger), 'wb')
    pickle.dump(output, outfile)
    outfile.close()
    # np.savetxt(
    #   'results/{}/run{}/result_{}_{}.csv'.format(args.flavor, args.run, i_event, i_trigger),
    #   results,
    #   delimiter=', '
    # )
    peaks = peakFinder.find_peaks(
      np.abs(scipy.signal.hilbert(np.sum(corr, axis=(0, 1)))),
      .2
    )
    plotting.plot_correlation(
      i_event,
      i_trigger,
      times, 
      corr,
      wf,
      wf_noiseless,
      probs,
      noise_rms,
      args.flavor,
      args.run,
      shower_signal_times,
      shower_energies,
      shower_had_fracs,
      peaks
    )
    plotting.plot_found_pulses(
      i_event,
      i_trigger,
      times, 
      corr,
      wf,
      wf_noiseless,
      probs,
      noise_rms,
      args.flavor,
      args.run,
      peaks
    )
    gc.collect()
