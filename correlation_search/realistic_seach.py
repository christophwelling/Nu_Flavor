import numpy as np
import matplotlib.pyplot as plt
import json
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
import os

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("--run", type=int, default=-1)
args = parser.parse_args()

upsampling_factor = 10.
n_samples = int(1024*upsampling_factor)
beamforming_degrees = 50.
antPosFile = os.environ['PUEO_UTIL_INSTALL_DIR'] + '/share/pueo/geometry/jun25/qrh.dat'
filter_band = [.2, 1.]
probability_thresholds = np.array([.05, 1.e-2, 1.e-3])
# filter_band = None
folders = []
if args.run < 0:
  folders = glob.glob(args.path + '/run*')
else:
  folders = [args.path + '/run{}'.format(args.run)]
  if not os.path.exists(folders[0]):
    raise RuntimeError("Path {} does not exist!".format(folders[0]))

templateHelper = helpers.template_helper.templateHelper(
  '../templates/templates.csv',
  'background_correlations.csv',
  upsampling_factor,
  filter_band,
  True
)

pulseFinder = helpers.pulse_finder.pulseFinder(
  antPosFile,
  upsampling_factor,
  filter_band,
  templateHelper,
  'background_correlations.csv'
)

for i_file, folder_name in enumerate(folders):
  run_id = int(folder_name.split('/')[-1][3:])
  filename = folder_name + '/IceFinal_{}_allTree.root'.format(run_id)
  output_filename = 'pulse_search_results_{}.json'.format(run_id)
  if os.path.exists(output_filename):
    continue
  print(i_file, folder_name)
  dataReader = helpers.data_reader.DataReader(
    filename,
    antPosFile,
    1,
    None
  )
  output = {'events': []}
  for i_event in range(dataReader.get_n_events()):
    dataReader.read_event(i_event)
    rf_dir = dataReader.get_signal_direction()
    max_channel = dataReader.get_max_channel()
    trigger_times = dataReader.get_trigger_times()
    times = dataReader.get_times()
    channel_ids = dataReader.get_close_channels(
      beamforming_degrees * np.pi / 180.,
      max_channel[0]
    )
    event_output = {
      'nu_energy': dataReader.get_neutrino_energy(),
      'absorption_weight': dataReader.get_neutrino_absorption_weight(),
      'sub_events': []
    }
    wf_nl = np.zeros((2, dataReader.get_n_samples()))
    wf_max = np.zeros((2, dataReader.get_n_samples()))
    wf_nl[0] = dataReader.get_waveform(
      max_channel[0],
      0,
      True
    )
    wf_nl[1] = dataReader.get_waveform(
      max_channel[0],
      1,
      True
    )
    wf_max[0] = dataReader.get_waveform(
      max_channel[0],
      0,
      False
    )
    wf_max[1] = dataReader.get_waveform(
      max_channel[0],
      1,
      False
    )

    fig0, axes0 = plt.subplots(2, 1, figsize=(16, 12))
    wf = np.zeros((2, channel_ids.shape[0], dataReader.get_n_samples()))
    for i_pol in range(2):
      axes0[i_pol].plot(
        times,
        wf_max[i_pol],
        color='C0'
      )
      axes0[i_pol].plot(
        times,
        wf_nl[i_pol],
        color='C1',
        alpha=.5
      )
      axes0[i_pol].grid()
      for i_ch in range(channel_ids.shape[0]):
        wf[i_pol, i_ch] = dataReader.get_waveform(
          channel_ids[i_ch],
          i_pol,
          False
        )
    for i_trig, trigger_time in enumerate(trigger_times):
      if trigger_time < 0:
        continue
      i_peak = np.argmin(np.abs(times - trigger_time))
      if i_peak < 512:
        i_peak = 512
      if i_peak > dataReader.get_n_samples() - 512:
        i_peak = dataReader.get_n_samples() - 512      
      sub_event = {
        'i_trigger': i_trig,
        'trigger_time': trigger_time,
        'pulses_found': []
      }
      waveforms = np.copy(wf[:, :, i_peak-512:i_peak+512])
      start_time = times[i_peak-512]
      pulseFinder.set_waveforms(
        waveforms,
        channel_ids
      )
      beams = pulseFinder.beamform(
        rf_dir,
        max_channel[0]
      )
      dedispersed_waveforms = pulseFinder.dedisperse()
      dedispersed_beams = pulseFinder.beamform(
        rf_dir,
        max_channel[0],
        True
      )
      correlations, probabilities = pulseFinder.correlate(
        dedispersed_beams,
        max_channel[1]
      )
      found_pulses = pulseFinder.get_pulses(
        correlations[max_channel[1]],
        probabilities[max_channel[1]],
        probability_thresholds
      )
      tt = pulseFinder.get_times()
      plt.close('all')
      fig1, axes1 = plt.subplots(5, 2, figsize=(24, 8)) 
      # fig2, axes2 = plt.subplots(4, 1, figsize=(12, 12)) 
      for i_pol in range(2):
        axes1[0, i_pol].plot(
          times - start_time,
          wf_max[i_pol],
          color='C0'
        )
        axes1[0, i_pol].plot(
          times - start_time,
          wf_nl[i_pol],
          color='C1',
          alpha=.8
        )
        axes1[0, i_pol].grid()
        axes1[0, i_pol].set_xlim([tt[0], tt[-1]])
        axes1[1, i_pol].plot(
          tt,
          beams[i_pol],
          color='C0',
          linewidth=1
        )
        # axes1[1, i_pol].plot(
        #   times - start_time,
        #   wf_nl[i_pol],
        #   color='C1',
        #   alpha=.5
        # )
        axes1[1, i_pol].grid()
        axes1[1, i_pol].set_xlim([tt[0], tt[-1]])
        axes0[i_pol].axvline(
          trigger_time,
          color='r',
          linestyle=':'
          )
        axes1[2, i_pol].plot(
          tt,
          dedispersed_beams[i_pol],
          color='C0'
        )
        axes1[2, i_pol].set_xlim([tt[0], tt[-1]])
        tt_correlations = np.arange(correlations.shape[1]) / 3. / upsampling_factor - (correlations.shape[1] - beams.shape[1]) / 3. / upsampling_factor
        # print((correlations.shape[0] - beams.shape[0]))
        axes1[2, i_pol].grid()
        axes1[3, i_pol].plot(
          tt_correlations,
          correlations[i_pol]
        )
        axes1[3, i_pol].grid()
        axes1[3, i_pol].set_xlim([tt[0], tt[-1]])
        axes1[4,i_pol].plot(
          tt_correlations,
          probabilities[i_pol]
        )
        axes1[4, i_pol].grid()
        axes1[4, i_pol].set_yscale('log')
        axes1[4, i_pol].set_ylim([1.e-4, 2.])
        axes1[4, i_pol].set_xlim([tt[0], tt[-1]])
        axes1[4, i_pol].set_xlabel('t [ns]')
        axes1[4, i_pol].set_ylabel('probability')
        axes1[0, i_pol].get_xaxis().set_ticklabels([])
        axes1[0, i_pol].set_ylabel('U [V]')
        axes1[1, i_pol].set_ylabel('U [V]')
        axes1[2, i_pol].set_ylabel('U [V]')
        axes1[3, i_pol].set_ylabel('correlation [a.u.]')
        axes1[1, i_pol].get_xaxis().set_ticklabels([])
        axes1[2, i_pol].get_xaxis().set_ticklabels([])
        axes1[3, i_pol].get_xaxis().set_ticklabels([])

        if i_pol == max_channel[1]:
          for i_pulse in range(found_pulses.shape[0]):
            axes1[4, i_pol].axvspan(
              tt_correlations[found_pulses[i_pulse, 0]],
              tt_correlations[found_pulses[i_pulse, 1]],
              color='C{}'.format(found_pulses[i_pulse, 2]),
              alpha=.2
            )
            axes1[4, i_pol].axvline(
              (tt_correlations[found_pulses[i_pulse, 0]] + tt_correlations[found_pulses[i_pulse, 1]]) * .5,
              color='C{}'.format(found_pulses[i_pulse, 2]),
              linestyle='--'
            )
            pulse_dict = {
              'i_pulse': int(i_pulse),
              'pulse_time': (tt_correlations[found_pulses[i_pulse, 0]] + tt_correlations[found_pulses[i_pulse, 1]]) * .5,
              'threshold': int(found_pulses[i_pulse, 2]),
              'min_probability': np.min(probabilities[i_pol, found_pulses[i_pulse, 0]: found_pulses[i_pulse, 1]])
            }
            sub_event['pulses_found'].append(pulse_dict)
      event_output['sub_events'].append(sub_event)
      fig1.tight_layout()
      fig1.savefig('plots/realistic_search/cut_waveforms/event_{}_{}_{}.png'.format(run_id, i_event, i_trig))
      # fig2.tight_layout()
      # fig2.savefig('plots/realistic_search/correlations/event_{}_{}_{}.png'.format(run_id, i_event, i_trig))
    output['events'].append(event_output)
    fig0.tight_layout()
    fig0.savefig('plots/realistic_search/full_waveforms/event_{}_{}.png'.format(run_id, i_event))
  json.dump(
    output,
    open(output_filename, 'w')
  )