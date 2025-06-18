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
parser.add_argument("--run", type=int, default=-1)
args = parser.parse_args()

upsampling_factor = 10.
n_samples = int(1024*upsampling_factor)
beamforming_degrees = 40.
antPosFile = '/home/welling/Software/pueo/usr/share/pueo/geometry/jun25/qrh.dat'
filter_band = [.2, 1.]
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
  'beamformed_corr_thresholds.csv',
  upsampling_factor,
  filter_band,
  True
)

pulseFinder = helpers.pulse_finder.pulseFinder(
  antPosFile,
  upsampling_factor,
  filter_band,
  templateHelper,
  'max_corr_beamformed.csv'
)

for i_file, folder_name in enumerate(folders):
  run_id = int(folder_name.split('/')[-1][3:])
  filename = folder_name + '/IceFinal_{}_allTree.root'.format(run_id)
  dataReader = helpers.data_reader.DataReader(
    filename,
    antPosFile,
    1,
    None
  )
  for i_event in range(dataReader.get_n_events()):
    print('Event ', i_event)
    dataReader.read_event(i_event)
    rf_dir = dataReader.get_signal_direction()
    max_channel = dataReader.get_max_channel()
    trigger_times = dataReader.get_trigger_times()
    times = dataReader.get_times()
    channel_ids = dataReader.get_close_channels(
      beamforming_degrees * np.pi / 180.,
      max_channel[0]
    )
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
      i_peak = np.argmin(np.abs(times - trigger_time))
      if i_peak < 512:
        i_peak = 512
      if i_peak > dataReader.get_n_samples() - 512:
        i_peak = dataReader.get_n_samples() - 512      
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
      tt = pulseFinder.get_times()
      plt.close('all')
      fig1, axes1 = plt.subplots(4, 1, figsize=(12, 12)) 
      fig2, axes2 = plt.subplots(4, 1, figsize=(12, 12)) 
      for i_pol in range(2):
        axes1[i_pol].plot(
          tt,
          beams[i_pol],
          color='C0',
          linewidth=1
        )
        axes1[i_pol].plot(
          times - start_time,
          wf_nl[i_pol],
          color='C1',
          alpha=.5
        )
        axes1[i_pol].grid()
        axes1[i_pol].set_xlim([120, 220])
        # axes1[i_pol+2].set_xlim([100, 200])
        axes1[i_pol].set_xlim([tt[0], tt[-1]])
        axes0[i_pol].axvline(
          trigger_time,
          color='r',
          linestyle=':'
          )
        axes1[i_pol+2].plot(
          tt,
          dedispersed_beams[i_pol],
          color='C0'
        )
        axes1[i_pol+2].grid()
        axes2[i_pol].plot(
          tt,
          correlations[i_pol]
        )
        axes2[i_pol].grid()
        axes2[i_pol+2].plot(
          tt,
          probabilities[i_pol]
        )
        axes2[i_pol+2].grid()
        axes2[i_pol+2].set_yscale('log')
        axes2[i_pol+2].set_ylim([1.e-4, 1.1])
      fig1.tight_layout()
      fig1.savefig('plots/realistic_search/cut_waveforms/event_{}_{}_{}.png'.format(run_id, i_event, i_trig))
      fig2.tight_layout()
      fig2.savefig('plots/realistic_search/correlations/event_{}_{}_{}.png'.format(run_id, i_event, i_trig))
    
    fig0.tight_layout()
    fig0.savefig('plots/realistic_search/full_waveforms/event_{}_{}.png'.format(run_id, i_event))