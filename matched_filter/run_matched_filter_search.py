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
trigger_pos = int(280 * upsampling_factor)
mf_helper = matched_filter_helper.MatchedFilterHelper(
  '/project/avieregg/welling/pueo/flavor/noise/run99/IceFinal_99_allTree.root',
  upsampling_factor

)
mf_helper.calculate_noise_spectral_density()
peakFinder = helpers.peakfinder.PeakFinder()

polEstimator = helpers.polarization_estimator.polarizationEstimator()

folders = []
print('Path ', args.path)
if args.run < 0:
  folders = glob.glob(args.path+'*')
else:
  folders = [args.path + '/run{}'.format(args.run)]
print('Path: ', args.path)
print('Folders: ', folders)
for i_file, folder_name in enumerate(folders):
  run_id = int(folder_name.split('/')[-1][3:])
  flavor = folder_name.split('/')[-2]
  filename = folder_name + '/IceFinal_{}_allTree.root'.format(run_id)
  print('run_id: ', run_id)
  dataReader = helpers.data_reader.DataReader(
    filename,
    None,
    upsampling_factor
  )
  for i_event in range(min(10000, dataReader.get_n_events())):
    dataReader.read_event(i_event)
    print('---------->>>  Event ', i_event, '<<<----------')
    signal_direction = dataReader.get_signal_direction()
    if not os.path.isdir('/project/avieregg/welling/pueo/flavor/found_pulses/{}/run{}/'.format(flavor, run_id)):
      os.makedirs('/project/avieregg/welling/pueo/flavor/found_pulses/{}/run{}/'.format(flavor, run_id))
    found_pulses_filename = '/project/avieregg/welling/pueo/flavor/found_pulses/{}/run{}/pulses_{}.json'.format(flavor, run_id, i_event)
    if os.path.isfile(found_pulses_filename):
      continue
    event_output = {
        'nu_energy': dataReader.get_neutrino_energy(),
        'weight': dataReader.get_event_weight(),
        'inelasticity': dataReader.get_inelasticity(),
        'viewing_angle': dataReader.get_viewing_angle(),
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
    print('trigger times: ', dataReader.get_trigger_times())
    for i_trigger, trigger_time in enumerate(dataReader.get_trigger_times()):
      print('---------------------')
      if trigger_time <= 0:
        print('Negative trigger time {}, skipping event'.format(trigger_time))
        continue
      trigger_index = np.argmin(np.abs(trigger_time - times_))
      trigger_shift = trigger_pos
      if trigger_index < trigger_pos:
        trigger_index = trigger_pos
        trigger_shift = 512 * upsampling_factor
      if trigger_index > wf_.shape[2] - (n_samples - trigger_pos):
        trigger_index = wf_.shape[2] - (n_samples - trigger_pos)
        trigger_shift = 512 * upsampling_factor
      sub_event = {
        'i_trigger': i_trigger,
        'trigger_time': trigger_time,
        'pulses_found': []
      }
      wf = wf_[:, :, trigger_index-trigger_pos:trigger_index+(n_samples - trigger_pos)]
      wf_noiseless = wf_noiseless_[:, :, trigger_index-trigger_pos: trigger_index+(n_samples - trigger_pos)]
      shower_signal_times = dataReader.get_det_times() * 1.e9  - trigger_time + trigger_shift/upsampling_factor/3.+215
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
        5000
      )
      # if not os.path.isdir('results/{}/run{}'.format(flavor, run_id)):
      #   os.makedirs('results/{}/run{}'.format(flavor, run_id))
      results = np.zeros((6, times.shape[0]))
      results[:2] = wf[:, max_channel] / noise_rms
      results[2:4] = wf_noiseless[:, max_channel] / noise_rms
      results[4] = np.sum(corr, axis=(0, 1))
      results[5] = probs
      # output = {
      #   'data': results,
      #   'shower_times': shower_signal_times,
      #   'shower_energies': shower_energies,
      #   'shower_hadfract': shower_had_fracs
      # }
      # outfile = open('results/{}/run{}/result_{}_{}.pkl'.format(flavor, run_id, i_event, i_trigger), 'wb')
      # pickle.dump(output, outfile)
      # outfile.close()
      # np.savetxt(
      #   'results/{}/run{}/result_{}_{}.csv'.format(args.flavor, args.run, i_event, i_trigger),
      #   results,
      #   delimiter=', 'fla
      # )
      peaks = peakFinder.find_peaks(
        np.abs(scipy.signal.hilbert(np.sum(corr, axis=(0, 1)))),
        np.sum(corr, axis=(0, 1)),
        .3
      )
      n_pulses = 0
      signal_directions = dataReader.get_signal_directions()
      max_angle_diff = 0
      for ii in range(signal_directions.shape[0]):
        for jj in range(ii, signal_directions.shape[0]):
          dot_prod = (np.dot(signal_directions[ii], signal_directions[jj]))
          angle = np.arccos(min(1, dot_prod))
          if angle > max_angle_diff:
            max_angle_diff = angle
      print('max angle: ', max_angle_diff * 180. / np.pi)
      for i_peak, peak in enumerate(peaks):
        if np.min(probs[peak[0]:peak[1]]) < threshold:
          peak_time = times[int(peak[0]+.5*(peak[1]-peak[0]))]
          i_closest_pulse = np.argmin(np.abs(peak_time - shower_signal_times))
          pulse_dir = signal_directions[i_closest_pulse]
          pulse_time_offset = peak_time - shower_signal_times[i_closest_pulse]
          sub_event['pulses_found'].append({
            'i_pulse': n_pulses,
            'pulse_time': peak_time,
            'threshold': threshold,
            'min_probability': np.min(probs[peak[0]:peak[1]]),
            'max_corr': np.max(np.sum(corr, axis=(0, 1))[peak[0]:peak[1]]),
            'mc_signal_dir': list(pulse_dir),
            'mc_signal_time_offset': pulse_time_offset
          })
          print('MC pulse time offset:', pulse_time_offset)
          n_pulses += 1
          if n_pulses > 1:
            max_ang = 0
            for jj in range(n_pulses-1):
              angle_diff = np.arccos(min(1, np.dot(sub_event['pulses_found'][jj]['mc_signal_dir'], sub_event['pulses_found'][n_pulses-1]['mc_signal_dir'])))
              if angle_diff > max_ang:
                max_ang = angle_diff
            print('Pulse angle offset: ', max_ang * 180. / np.pi)
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
      # plotting.plot_found_pulses(
      #   i_event,
      #   i_trigger,
      #   times, 
      #   corr,
      #   wf,
      #   wf_noiseless,
      #   probs,
      #   noise_rms,
      #   flavor,
      #   run_id,
      #   peaks
      # )
      gc.collect()
    json.dump(
      event_output,
      open(found_pulses_filename, 'w')
    )
