import numpy as np
import sys
import gc
sys.path.append('../')
import helpers.data_reader
import matplotlib.pyplot as plt
import matched_filter_helper
import argparse
import plotting


parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
args = parser.parse_args()

antenna_angle_cut = 50. * np.pi / 180.

dataReader = helpers.data_reader.DataReader(
  args.filename,
  None
)
mf_helper = matched_filter_helper.MatchedFilterHelper()


for i_event in range(dataReader.get_n_events()):
  print(i_event, ' / ', dataReader.get_n_events())
  dataReader.read_event(i_event)
  signal_direction = dataReader.get_signal_direction()
  antennas = mf_helper.get_antenna_indices(signal_direction, antenna_angle_cut)
  times_ = dataReader.get_times()
  times = np.arange(1024) / 3.
  wf_ = np.zeros((2, len(antennas), times_.shape[0]))
  wf_noiseless_ = np.zeros_like(wf_)
  polarization_angle = dataReader.get_polarization_angle()
  viewing_angle = dataReader.get_viewing_angle()
  for i_ant, ant in enumerate(antennas):
    wf_[0, i_ant] = dataReader.get_waveform(ant, 0)
    wf_[1, i_ant] = dataReader.get_waveform(ant, 1)
    wf_noiseless_[0, i_ant] = dataReader.get_waveform(ant, 0, True)
    wf_noiseless_[1, i_ant] = dataReader.get_waveform(ant, 1, True)
  for i_trigger, trigger_time in enumerate(dataReader.get_trigger_times()):
    if trigger_time <= 0:
      continue
    trigger_index = np.argmin(np.abs(trigger_time - times_))
    if trigger_index < 512:
      trigger_index = 512
    if trigger_index > wf_.shape[2] - 512:
      trigger_index = wf_.shape[2] - 512
    wf = wf_[:, :, trigger_index-512:trigger_index+512]
    wf_noiseless = wf_noiseless_[:, :, trigger_index-512: trigger_index+512]
    plotting.plot_waveforms(
      i_event,
      i_trigger,
      times,
      wf,
      wf_noiseless
    )
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
    plotting.plot_correlation(
      i_event,
      i_trigger,
      times, 
      corr,
      wf,
      wf_noiseless
    )
    gc.collect()

el = -10. * np.pi / 180.
signal_direction = np.array([np.cos(el), 0, np.sin(el)])
antennas = mf_helper.get_antenna_indices(signal_direction, 30.*np.pi / 180.)
template = mf_helper.generate_template(
  signal_direction,
  antennas,
  0.1
)
template_time_domain = np.fft.irfft(template, axis=2)
freqs = np.fft.rfftfreq(1024, 1./3.)
times = np.arange(1024) / 3.
n_sectors = len(antennas) // 4
fig1, ax1 = plt.subplots(8, n_sectors, figsize=(n_sectors*4, 12))
for i_sector in range(n_sectors):
  for i_ring in range(4):
    ax1[i_ring, i_sector].plot(
      times,
      template_time_domain[0, i_sector*4+i_ring]
    )
    ax1[i_ring, i_sector].plot(
      times,
      template_time_domain[1, i_sector*4+i_ring]
    )
    ax1[i_ring+4, i_sector].plot(
      freqs,
      np.abs(template[0, i_sector*4+i_ring])
    )
    ax1[i_ring+4, i_sector].plot(
      freqs,
      np.abs(template[1, i_sector*4+i_ring])
    )
    ax1[i_ring, i_sector].grid()
    ax1[i_ring, i_sector].set_xlim([0, 100])
    ax1[i_ring+4, i_sector].grid()
    max_td = np.max(np.abs(template_time_domain))
    max_fd = np.max(np.abs(template))
    ax1[i_ring, i_sector].set_ylim([-1.1*max_td, 1.1*max_td])
    ax1[i_ring+4, i_sector].set_ylim([0, 1.1*max_fd])
fig1.tight_layout()
fig1.savefig('template.png')
