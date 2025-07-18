import numpy as np
import sys
sys.path.append('../')
import os
import helpers.antenna_helper
import NuRadioMC.SignalGen.askaryan

class MatchedFilterHelper:
  def __init__(
      self
  ):
    self.__antenna_helper = helpers.antenna_helper.AntennaHelper()
    amp_data = np.genfromtxt(
      os.environ['PUEO_UTIL_INSTALL_DIR'] + '/share/pueo/responses/signalChainMI/PUEO_SignalChainMI_0.csv',
      delimiter=','
    )
    self.__signal_chain = amp_data[:, 1] * np.exp(1.j * amp_data[:, 2])
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