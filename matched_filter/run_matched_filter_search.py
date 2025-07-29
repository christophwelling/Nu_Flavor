import numpy as np
import sys
import gc
import os
import glob
import pickle
sys.path.append('../')
import helpers.data_reader
import helpers.peakfinder
import matched_filter_helper
import argparse
import plotting
import scipy.signal
import helpers.polarization_estimator
import radiotools.helper
import json


parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('--run', type=int, default=-1)
args = parser.parse_args()

antenna_angle_cut = 50. * np.pi / 180.
threshold = .05
elevation_error = .15 * np.pi / 180.
azimuth_error = .5 * np.pi / 180.
simulate_direction_errors = True

upsampling_factor = 1
n_samples = int(1024 * upsampling_factor)

mf_helper = matched_filter_helper.MatchedFilterHelper(
  '~/RadioNeutrino/data/pueo/flavor/noise/run99/IceFinal_99_allTree.root',
  upsampling_factor

)
mf_helper.calculate_noise_spectral_density()
peakFinder = helpers.peakfinder.PeakFinder()
polEstimator = helpers.polarization_estimator.polarizationEstimator()

folders = []
if args.run < 0:
  folders = glob.glob(args.path)
else:
  folders = [args.path + '/run{}'.format(args.run)]



for i_file, folder_name in enumerate(folders):
  run_id = int(folder_name.split('/')[-1][3:])
  flavor = folder_name.split('/')[-2]
  filename = folder_name + '/IceFinal_{}_allTree.root'.format(run_id)
  print(flavor)
  dataReader = helpers.data_reader.DataReader(
    filename,
    None,
    upsampling_factor
  )

  for i_event in range(dataReader.get_n_events()):
    print(i_event, ' / ', dataReader.get_n_events())
    dataReader.read_event(i_event)
    signal_direction = dataReader.get_signal_direction()
    if not os.path.isdir('found_pulses/{}/run{}/'.format(flavor, run_id)):
      os.makedirs('found_pulses/{}/run{}/'.format(flavor, run_id))
    found_pulses_filename = 'found_pulses/{}/run{}/pulses_{}.json'.format(flavor, run_id, i_event)
    if os.path.isfile(found_pulses_filename):
      continue
    event_output = {
        'nu_energy': dataReader.get_neutrino_energy(),
        'weight': dataReader.get_event_weight(),
        'sub_events': []
      }
    if simulate_direction_errors:
      theta, phi = radiotools.helper.cartesian_to_spherical(*signal_direction)
      theta += np.random.normal(0, elevation_error)
      phi += np.random.normal(0, azimuth_error)
      signal_direction = radiotools.helper.spherical_to_cartesian(theta, phi)
    antennas = mf_helper.get_antenna_indices(signal_direction, antenna_angle_cut)
    times_ = dataReader.get_times()
    times = np.arange(n_samples) / 3. / upsampling_factor
    wf_ = np.zeros((2, len(antennas), times_.shape[0]))
    wf_noiseless_ = np.zeros_like(wf_)
    polarization_angle = dataReader.get_polarization_angle()
    viewing_angle = dataReader.get_viewing_angle()
    if np.abs(viewing_angle - np.arccos(1./1.79)) > 10. * np.pi / 180.:
      viewing_angle =  viewing_angle + np.arccos(1./1.79) - np.arccos(1./1.325)
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
      sub_event = {
        'i_trigger': i_trigger,
        'trigger_time': trigger_time,
        'pulses_found': []
      }
      wf = wf_[:, :, trigger_index-n_samples//2:trigger_index+n_samples//2]
      wf_noiseless = wf_noiseless_[:, :, trigger_index-n_samples//2: trigger_index+n_samples//2]
      shower_signal_times = dataReader.get_det_times() * 1.e9 - times_[trigger_index - n_samples//2] + 215
      max_channel = np.argmax(np.max(wf_noiseless, axis=(0, 2)))

      rec_polarization_angle = polEstimator.estimate_polarization_angle(
        wf,
        mf_helper,
        signal_direction,
        antennas,
        viewing_angle
      )
      template = mf_helper.generate_template(
        signal_direction,
        antennas,
        rec_polarization_angle,
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
        1500
      )
      if not os.path.isdir('results/{}/run{}'.format(flavor, run_id)):
        os.makedirs('results/{}/run{}'.format(flavor, run_id))
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
      outfile = open('results/{}/run{}/result_{}_{}.pkl'.format(flavor, run_id, i_event, i_trigger), 'wb')
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
      n_pulses = 0
      for i_peak, peak in enumerate(peaks):
        if np.min(probs[peak[0]:peak[1]]) < threshold:
          sub_event['pulses_found'].append({
            'i_pulse': n_pulses,
            'pulse_time': times[int(peak[0]+.5*(peak[1]-peak[0]))],
            'threshold': threshold,
            'min_probability': np.min(probs[peak[0]:peak[1]])
          })
          n_pulses += 1
      event_output['sub_events'].append(sub_event)
      plotting.plot_correlation(
        i_event,
        i_trigger,
        times, 
        corr,
        wf,
        wf_noiseless,
        probs,
        noise_rms,
        flavor,
        run_id,
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
        flavor,
        run_id,
        peaks
      )
      gc.collect()
    json.dump(
      event_output,
      open(found_pulses_filename, 'w')
    )