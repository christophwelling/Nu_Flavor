import numpy as np
import os

class DetectorModel:
  def __init__(
      self,
      n_samples=1024,
      antenna_shift=0,
      amp_shift=0
  ):
    pueosim_dir = os.getenv('PUEO_UTIL_INSTALL_DIR')
    antenna_response_data = np.genfromtxt(
      pueosim_dir + '/share/pueo/responses/antennas/derived/toyon_vv_0.csv'
    )
    amp_response_data = np.genfromtxt(
      pueosim_dir + '/share/pueo/responses/signalChainMI/PUEO_SignalChainMI_0.csv',
      delimiter=','
    )
    antenna_response = antenna_response_data[:, 1] * np.exp(1.j * antenna_response_data[:, 2])
    if antenna_shift != 0:
      antenna_response = np.fft.rfft(
        np.roll(
          np.fft.irfft(antenna_response),
          antenna_shift
        )
      )
    amp_response = amp_response_data[:, 1] * np.exp(1.j * amp_response_data[:, 2])
    if amp_shift != 0:
      amp_response = np.fft.rfft(
        np.roll(
          np.fft.irfft(amp_response),
          amp_shift
        )
      )
    if antenna_response.shape[0] == n_samples // 2 + 1:
      self.__antenna_response = antenna_response
    elif antenna_response.shape[0] > n_samples // 2 + 1:
      self.__antenna_response = np.fft.rfft(
        np.fft.irfft(antenna_response)[:n_samples]
      )
    else:
      t_domain = np.zeros(n_samples)
      t_domain[:antenna_response.shape[0]*2 - 2] = np.fft.irfft(antenna_response)
      self.__antenna_response = np.fft.rfft(t_domain)
    
    if amp_response.shape[0] == n_samples // 2 + 1:
      self.__amp_response = amp_response
    elif amp_response.shape[0] > n_samples // 2 + 1:
      self.__amp_response = np.fft.rfft(
        np.fft.irfft(amp_response)[:n_samples]
      )
    else:
      t_domain = np.zeros(n_samples)
      t_domain[:amp_response.shape[0]*2 - 2] = np.fft.irfft(amp_response)
      self.__antenna_response = np.fft.rfft(t_domain)




  def get_amp_response(self):
    return self.__amp_response

  def get_antenna_response(self):
    return self.__antenna_response
  
  def get_instrument_response(self):
    return self.__amp_response * self.__antenna_response
