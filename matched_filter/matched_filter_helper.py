import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import os
import helpers.antenna_helper
import NuRadioMC.SignalGen.askaryan
import helpers.data_reader

class MatchedFilterHelper:
  def __init__(
      self,
      noise_data
  ):
    self.__antenna_helper = helpers.antenna_helper.AntennaHelper()
    amp_data = np.genfromtxt(
      os.environ['PUEO_UTIL_INSTALL_DIR'] + '/share/pueo/responses/signalChainMI/PUEO_SignalChainMI_0.csv',
      delimiter=','
    )
    self.__signal_chain = amp_data[:, 1] * np.exp(1.j * amp_data[:, 2])
    self.__noise_reader = helpers.data_reader.DataReader(
      noise_data,
      None
      )
  def generate_template(
      self,
      signal_direction: np.array,
      antenna_indices: list,
      polarization_angle: float,
      viewing_angle=1.*np.pi / 180.
  ):
    efield  =NuRadioMC.SignalGen.askaryan.get_frequency_spectrum(
      1.e18,
      viewing_angle,
      1024,
      1./3.,
      'HAD',
      1.78, 
      1000.,
      'Alvarez2009'
    )
    template = np.zeros((2, len(antenna_indices), 513), dtype=complex)
    for i_pol in range(2):
      for i_antenna, antenna_index in enumerate(antenna_indices):
        template[i_pol, i_antenna] = self.__antenna_helper.get_antenna_response_for_antenna(
          signal_direction,
          antenna_index,
          i_pol,
          True
        ) * efield
    template[0] *= np.cos(polarization_angle)
    template[1] *= np.sin(polarization_angle)
    return template * self.__signal_chain

  def get_antenna_indices(
      self,
      signal_direction,
      max_azimuth_difference
  ):
    antenna_boresights = self.__antenna_helper.get_antenna_boresights()
    antenna_boresights[:, 2] = 0
    antenna_boresights = (antenna_boresights.T / np.sqrt(np.sum(antenna_boresights**2, axis=1))).T
    signal_dir = np.copy(signal_direction)
    signal_dir[2] = 0
    signal_dir /= np.sqrt(np.sum(signal_dir**2))
    indices = np.arange(antenna_boresights.shape[0], dtype=int)
    # print(np.dot(-signal_dir, antenna_boresights.T), np.cos(max_azimuth_difference))
    return indices[np.dot(-signal_dir, antenna_boresights.T) > np.cos(max_azimuth_difference)]

  def apply_matched_filter(
      self,
      template,
      data
  ):
    data_fd = np.fft.rfft(data, axis=2)
    filtered_data = template * data_fd.conjugate()
    return np.roll(np.fft.irfft(filtered_data, axis=2), 380)

  def estimate_background_rate(
      self,
      template,
      channel_indices,
      corrs,
      n_events=500
  ):
    wf = np.zeros((2, len(channel_indices), 1024))
    max_corr = np.zeros(min(n_events, self.__noise_reader.get_n_events()))
    noise_rms = np.zeros_like(max_corr)
    for i_event in range(min(n_events, self.__noise_reader.get_n_events())):
      self.__noise_reader.read_event(i_event)
      for i_index, ch_index in enumerate(channel_indices):
        for i_pol in range(2):
          wf[i_pol, i_index] = self.__noise_reader.get_waveform(
            ch_index,
            i_pol,
            False
          )
      wf_filtered = self.apply_matched_filter(
        template,
        wf
      )
      noise_rms[i_event] = np.sqrt(np.mean(wf**2))
      max_corr[i_event] = np.max(np.abs(scipy.signal.hilbert(np.sum(np.sum(wf_filtered, axis=0), axis=0))))
    entries, bins, patches = plt.hist(
      max_corr,
      bins=100,
      density=True,
      cumulative=-1
    )
    plt.plot(bins[1:], entries)
    plt.close('all')
    corr_hilbert = np.abs(scipy.signal.hilbert(corrs))
    probabilities = np.zeros_like(corrs)
    for i_sample in range(probabilities.shape[0]):
      probabilities[i_sample] = entries[np.argmin(np.abs(corr_hilbert[i_sample] - bins[1:]))]
    return probabilities, np.mean(noise_rms)


