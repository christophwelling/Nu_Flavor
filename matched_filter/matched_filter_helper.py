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
      noise_data,
      upsampling_factor
  ):
    self.__antenna_helper = helpers.antenna_helper.AntennaHelper(upsampling_factor)
    self.__upsampling_factor = upsampling_factor
    self.__n_samples = int(1024 * upsampling_factor)
    amp_data = np.genfromtxt(
      os.environ['PUEO_UTIL_INSTALL_DIR'] + '/share/pueo/responses/signalChainMI/PUEO_SignalChainMI_0.csv',
      delimiter=','
    )
    signal_chain_td = scipy.signal.resample(np.fft.irfft(amp_data[:, 1] * np.exp(1.j * amp_data[:, 2])), self.__n_samples)
    self.__signal_chain = np.fft.rfft(signal_chain_td)
    self.__noise_reader = helpers.data_reader.DataReader(
      noise_data,
      None,
      upsampling_factor
      )
    self.__noise_power_spectrum = 1.
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
      self.__n_samples,
      1./3. / self.__upsampling_factor,
      'HAD',
      1.79, 
      1000.,
      'Alvarez2009'
    )
    template = np.zeros((2, len(antenna_indices), self.__n_samples // 2 + 1), dtype=complex)
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
    filtered_data = (template * data_fd.conjugate() / self.__noise_power_spectrum).conjugate()
    return -np.roll(np.fft.irfft(filtered_data, axis=2), int(-380 * self.__upsampling_factor))

  def estimate_background_rate(
      self,
      template,
      channel_indices,
      corrs,
      n_events=500
  ):
    wf = np.zeros((2, len(channel_indices), self.__n_samples))
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
      max_corr[i_event] = np.max(((np.sum(np.sum(wf_filtered, axis=0), axis=0))))
    entries, bins, patches = plt.hist(
      max_corr,
      bins=100,
      density=True,
      cumulative=-1
    )
    plt.figure(num=1, clear=True)
    plt.plot(bins[1:], entries)
    plt.close('all')
    # corr_hilbert = np.abs(scipy.signal.hilbert(corrs))
    probabilities = np.zeros_like(corrs)
    for i_sample in range(probabilities.shape[0]):
      probabilities[i_sample] = entries[np.argmin(np.abs(corrs[i_sample] - bins[1:]))]
    return probabilities, np.mean(noise_rms)

  def calculate_noise_spectral_density(
      self,
      n_events=100
  ):
    n_ev = min(n_events, self.__noise_reader.get_n_events())
    noise_power = np.zeros((n_ev, 2, 96, self.__n_samples // 2 + 1))
    for i_event in range(n_ev):
      self.__noise_reader.read_event(i_event)
      for i_ch in range(96):
        for i_pol in range(2):
          spec = np.fft.rfft(self.__noise_reader.get_waveform(i_ch, i_pol, False))
          noise_power[i_event, i_pol, i_ch] = np.abs(spec)**2
    self.__noise_power_spectrum = np.mean(noise_power, axis=(0, 1, 2))
    self.__noise_power_spectrum[self.__noise_power_spectrum < 5.e-3 * np.max(self.__noise_power_spectrum)] = 5.e-3 * np.max(self.__noise_power_spectrum)

