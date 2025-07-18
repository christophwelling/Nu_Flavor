import numpy as np
import os

class AntennaHelper:
  def __init__(
      self
  ):
    antenna_response_directory = os.environ['PUEO_UTIL_INSTALL_DIR'] + '/share/pueo/responses/antennas/derived/'
    self.__receiving_angles = np.array([0., 0.08726646, 0.17453293, 0.26179939, 0.34906585, 0.43633231, 0.52359878, 0.61086524, 0.6981317, 0.78539816, 0.87266463, 1.04719755, 1.22173048, 1.3962634, 1.57079633])
    self.__boresight_response = np.zeros((2, 513), dtype=complex)
    self.__boresight_response[0] = self.__read_boresight_response(antenna_response_directory + '/toyon_hh_0.csv')
    self.__boresight_response[1] = self.__read_boresight_response(antenna_response_directory + '/toyon_vv_0.csv')
    self.__off_axis_responses = np.zeros((2, 2, self.__receiving_angles.shape[0], 513), dtype=complex)
    self.__off_axis_responses[0, 0] = self.__read_off__axis_responses(antenna_response_directory + '/toyon_hh_az.csv')
    self.__off_axis_responses[0, 1] = self.__read_off__axis_responses(antenna_response_directory + '/toyon_hh_el.csv')
    self.__off_axis_responses[1, 0] = self.__read_off__axis_responses(antenna_response_directory + '/toyon_vv_az.csv')
    self.__off_axis_responses[1, 1] = self.__read_off__axis_responses(antenna_response_directory + '/toyon_vv_el.csv')

    position_data = np.genfromtxt(
      os.environ['PUEO_UTIL_INSTALL_DIR'] + '/share/pueo/geometry/jun25/qrh.dat',
      delimiter=',',
      skip_header=2
    )
    self.__antenna_positions = position_data[:, 2:5]
    self.__antenna_boresights = np.zeros_like(self.__antenna_positions)
    degrad = np.pi / 180.
    self.__antenna_boresights[:, 0] = np.cos(position_data[:, 6]*degrad) * np.cos(position_data[:, 5]*degrad)
    self.__antenna_boresights[:, 1] = np.sin(position_data[:, 6]*degrad) * np.cos(position_data[:, 5]*degrad)
    self.__antenna_boresights[:, 2] = np.sin(position_data[:, 5]*degrad)

  def __read_boresight_response(
      self,
      filename: str
  ):
    data = np.genfromtxt(
      filename
    )
    response = data[:, 1] * np.exp(1.j * data[:, 2])
    return response

  def __read_off__axis_responses(
      self,
      filename: str
  ):
    data = np.genfromtxt(
      filename
    )
    response = np.zeros((data.shape[0]//513, 513), dtype=complex)
    for i_angle in range(response.shape[0]):
      response[i_angle] = data[i_angle*513:i_angle*513+513, 1] * np.exp(1.j * data[i_angle*513:i_angle*513+513, 2])
    return response
  
  def get_antenna_response(
      self,
      azimuth_angle: float,
      elevation: float,
      polarization: int # 0: hpol, 1: vpol
  ):
    if polarization < 0 or polarization > 1:
      raise ValueError("Invalid value for polarization")
    
    azimuth_bin = np.argmin(np.abs(np.abs(azimuth_angle) - self.__receiving_angles))
    if self.__receiving_angles[azimuth_bin] > np.abs(azimuth_angle):
      azimuth_bin -= 1
    d_az = np.abs(np.abs(azimuth_angle) - self.__receiving_angles[azimuth_bin]) / np.abs(self.__receiving_angles[azimuth_bin+1] - self.__receiving_angles[azimuth_bin])
    az_correction = self.__off_axis_responses[polarization, 0, azimuth_bin] + d_az * (
      self.__off_axis_responses[polarization, 0, azimuth_bin+1] - self.__off_axis_responses[polarization, 0, azimuth_bin]
      )

    elevation_bin = np.argmin(np.abs(np.abs(elevation) - self.__receiving_angles))
    if self.__receiving_angles[elevation_bin] > np.abs(elevation):
      elevation_bin -= 1
    d_el = np.abs(np.abs(elevation) - self.__receiving_angles[elevation_bin]) / np.abs(self.__receiving_angles[elevation_bin+1] - self.__receiving_angles[elevation_bin])
    el_correction = self.__off_axis_responses[polarization, 1, elevation_bin] + d_el * (
      self.__off_axis_responses[polarization, 1, elevation_bin+1] - self.__off_axis_responses[polarization, 1, elevation_bin]
    )
    return self.__boresight_response[polarization] * az_correction * el_correction
  
  def get_antenna_response_for_direction(
      self,
      signal_direction: np.array,
      antenna_boresight: np.array,
      polarization: int # 0: hpol, 1: vpol
  ):
    if np.dot(signal_direction, antenna_boresight) > 0:
      return np.zeros(513)
    antenna_local_x = np.cross(antenna_boresight, np.array([0, 0, 1]))
    antenna_local_z = np.cross(antenna_boresight, antenna_local_x)
    elevation = np.arcsin(np.abs(np.dot(-signal_direction, antenna_local_z)))
    azimuth = np.arcsin(np.abs(np.dot(-signal_direction, antenna_local_x)))
    return self.get_antenna_response(
      azimuth,
      elevation,
      polarization
    )
  
  def get_antenna_response_for_antenna(
      self,
      signal_direction: np.array,
      antenna_index: int,
      polarization: int, #0: hpol, 1: vpol
      signal_travel_time: False # If true, the time shift due to the signal travel time is included in the antenna response
  ):
    response = self.get_antenna_response_for_direction(
      signal_direction,
      self.__antenna_boresights[antenna_index],
      polarization
    )
    if signal_travel_time:
      dt = np.dot(signal_direction, self.__antenna_positions[antenna_index]) / 3.e8
      freqs = np.fft.rfftfreq(1024, 1./3.e9)
      return response * np.exp(-2.j * np.pi * dt * freqs)
    else:
      return response
    
  def get_antenna_positions(self):
    return np.copy(self.__antenna_positions)
  
  def get_antenna_boresights(self):
    return np.copy(self.__antenna_boresights)